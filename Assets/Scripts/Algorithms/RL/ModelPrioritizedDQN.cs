using System;
using NN;
using NN.CPU_Single;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Algorithms.RL
{
    public class ModelPrioritizedDQN : ModelDQN
    {
        private readonly float _alpha;
        private readonly float _initialBeta;
        private float _beta;
        private readonly float[] _sampleWeights;
        private int _counter;

        private float _maxPriority;
        private int _maxPriorityIndex;
        private float _maxWeight;

        private readonly SumTree _sumTree;

        public ModelPrioritizedDQN(NetworkModel networkModel, NetworkModel targetModel,
            int numberOfActions,
            int stateSize, float beta = 0.4f, float alpha = 0.6f, int maxExperienceSize = 10000,
            int minExperienceSize = 100, int batchSize = 32,
            float gamma = 0.99f) : base(networkModel, targetModel, numberOfActions, stateSize, maxExperienceSize,
            minExperienceSize, batchSize, gamma)
        {
            _alpha = alpha;
            _beta = beta;
            _initialBeta = beta;
            _sampleWeights = new float[batchSize];
            _counter = 0;
            //_alpha = 0f;

            _maxPriority = 1.0f;
            _maxPriorityIndex = 0;

            _sumTree = new SumTree(maxExperienceSize);
            // _sumTree.UpdateValue(0, 1.0f);
        }

        public override void AddExperience(float[] currentState, int action, float reward, bool done, float[] nextState)
        {
            var experience = new Experience(currentState, action, reward, done, nextState);

            if (_experiences.Count < _maxExperienceSize)
            {
                _experiences.Add(experience);
            }
            else
            {
                _experiences[_lastExperiencePosition] = experience;
            }

            _sumTree.UpdateValue(_lastExperiencePosition, _maxPriority);
            //_sumTree.UpdateValue(_lastExperiencePosition, _sumTree.MaxValue());
            _lastExperiencePosition = (_lastExperiencePosition + 1) % _maxExperienceSize;
        }

        public override void Train()
        {
            if (_experiences.Count < _minExperienceSize)
                return;

            UpdateBeta();
            RandomBatch();

            // MaxByRow(_networkModel.Predict(_nextStates));
            // var targetPredictions = _targetModel.Predict(_nextStates);
            // NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));
            // for (int i = 0; i < _nextQ.Length; i++)
            // {
            //     var experience = _experiences[_batchIndexes[i]];
            //     _yTarget[i, experience.Action] =
            //         experience.Done
            //             ? experience.Reward
            //             : experience.Reward + _gamma * targetPredictions[i, _nextQ[i].index];
            // }


            //TODO: could do loss and priority update in this loop
            MaxByRow(_targetModel.Predict(_nextStates));
            NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));
            for (int i = 0; i < _batchSize; i++)
            {
                _sampleWeights[i] /= _maxWeight;

                var experience = _experiences[_batchIndexes[i]];
                _yTarget[i, experience.Action] =
                    experience.Done ? experience.Reward : experience.Reward + _gamma * _nextQ[i].value;
            }

            _networkModel.SetLossParams(_sampleWeights);
            UpdatePriorities(_networkModel.Loss(_yTarget));

            _networkModel.Update(_yTarget);
        }

        protected override void RandomBatch()
        {
            var totalExperiences = _experiences.Count;

            // if indexes can be repeated, then a multi threaded version might be faster
            var total = _sumTree.Total();
            _maxWeight = 0.0f;
            for (int i = 0; i < _batchSize; i++)
            {
                //var total = _sumTree.Total();
                //var batchIndex = _sumTree.Sample(Random.Range(1e-5f, total), out var priority);
                var batchIndex = _sumTree.Sample(Random.Range(0.0f, total), out var priority);

                if (batchIndex >= totalExperiences)
                {
                    // TODO: batchIndex = (_maxExperienceSize + _lastExperiencePosition - 1) % _maxExperienceSize;
                    batchIndex = totalExperiences - 1;
                    priority = _sumTree.Get(batchIndex);
                }
                //_sumTree.UpdateValue(batchIndex, 0f);

                _batchIndexes[i] = batchIndex;
                var experience = _experiences[batchIndex];
                for (int j = 0; j < _stateLenght; j++)
                {
                    _nextStates[i, j] = experience.NextState[j];
                    _currentStates[i, j] = experience.CurrentState[j];
                }

                var weight = Mathf.Pow(totalExperiences * (priority / total), -_beta);

                _sampleWeights[i] = weight;

                if (_maxWeight > weight) continue;

                _maxWeight = weight;
            }
        }

        private void UpdateBeta()
        {
            _counter++;
            var value = _initialBeta + _counter * (1.0f - _initialBeta) / 100000;
            _beta = 1.0f < value ? 1.0f : value;
        }

        private void UpdatePriorities(float[] samplePriorities)
        {
            for (int i = 0; i < _batchSize; i++)
            {
                //TODO: should do serious tests to see if this works better than the sqrt of the sample priorities
                var priority = Mathf.Pow(samplePriorities[i] + 1e-5f, _alpha);
                var priorityIndex = _batchIndexes[i];

                _sumTree.UpdateValue(priorityIndex, priority);

                if (_maxPriority >= priority) continue;

                _maxPriority = priority;
                _maxPriorityIndex = priorityIndex;
            }

            if (Math.Abs(_sumTree.Get(_maxPriorityIndex) - _maxPriority) == 0.0f) return;

            var totalExperiences = _experiences.Count;

            // _maxPriority = 0.0f;
            // for (int i = 0; i < _experiences.Count; i++)
            // {
            //     var priority = _sumTree.Get(i);
            //
            //     if (_maxPriority > priority) continue;
            //
            //     _maxPriority = priority;
            //     _maxPriorityIndex = i;
            // }

            var mid = totalExperiences / 2;
            _maxPriority = float.MinValue;
            _maxPriorityIndex = 0;
            for (int i = 0; i < mid; i++)
            {
                var priorityLeft = _sumTree.Get(i);
                var priorityRight = _sumTree.Get(i + mid);

                if (_maxPriority < priorityLeft)
                {
                    _maxPriority = priorityLeft;
                    _maxPriorityIndex = i;
                }

                if (_maxPriority > priorityRight) continue;

                _maxPriority = priorityRight;
                _maxPriorityIndex = i + mid;
            }

            var lastValue = _sumTree.Get(totalExperiences - 1);
            if (_maxPriority > lastValue) return;

            _maxPriority = lastValue;
            _maxPriorityIndex = totalExperiences - 1;
        }
    }
}