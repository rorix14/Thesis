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
        private readonly float[] _priorities;
        private readonly float[] _sampleWeights;
        private int _counter;

        private float _prioritiesSum;
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
            _priorities = new float[maxExperienceSize];
            _sampleWeights = new float[batchSize];
            _counter = 0;
            //_alpha = 0f;

            _prioritiesSum = 0.0f;
            _maxPriority = 1.0f;
            _maxPriorityIndex = 0;

            _sumTree = new SumTree(maxExperienceSize);
            //_sumTree = new SumTree((int)(maxExperienceSize / 0.6f));
        }

        public override void AddExperience(float[] currentState, int action, float reward, bool done, float[] nextState)
        {
            var experience = new Experience(currentState, action, reward, done, nextState);

            if (_experiences.Count < _maxExperienceSize)
            {
                _experiences.Add(experience);
                _prioritiesSum += _maxPriority;
            }
            else
            {
                _experiences[_lastExperiencePosition] = experience;
                // might not be needed
                _prioritiesSum += _maxPriority - _priorities[_lastExperiencePosition];
            }

            _sumTree.UpdateValue(_lastExperiencePosition, _maxPriority);

            _priorities[_lastExperiencePosition] = _maxPriority;
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
            //var total = _prioritiesSum;
            var total = _sumTree.Total();
            _maxWeight = 0.0f;
            for (int i = 0; i < _batchSize; i++)
            {
                var ran = Random.Range(0.0f, total);
                // var batchIndex = -1;
                // var cutoff = ran;
                // while (cutoff >= 0.0f)
                // {
                //     batchIndex++;
                //     cutoff -= _priorities[batchIndex];
                //
                //     if (batchIndex >= totalExperiences - 1) break;
                // }

                // Can just use the total function of the sum tree 

                var randomValue = ran;
                var batchIndex = _sumTree.Sample(randomValue, out var priority);

                if (batchIndex >= totalExperiences)
                {
                    // Debug.Log("Value: " + tt + ", lenght: " + totalExperiences + ", index: " + batchIndexx +
                    //           ", Total: " + total + ", random: " + randomValue + ", last value: " + _priorities[totalExperiences - 1]);
                    batchIndex = totalExperiences - 1;
                    priority = _sumTree.Get(batchIndex);
                }


                _batchIndexes[i] = batchIndex;
                var experience = _experiences[batchIndex];
                for (int j = 0; j < _stateLenght; j++)
                {
                    _nextStates[i, j] = experience.NextState[j];
                    _currentStates[i, j] = experience.CurrentState[j];
                }

                //var weight = Mathf.Pow(totalExperiences * (_priorities[batchIndex] / _prioritiesSum), -_beta);
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

                _prioritiesSum += priority - _priorities[priorityIndex];
                _priorities[priorityIndex] = priority;

                _sumTree.UpdateValue(priorityIndex, priority);

                if (_maxPriority >= priority) continue;

                _maxPriority = priority;
                _maxPriorityIndex = priorityIndex;
            }

            //TODO: Might be useful to have a max tree, to find the max instead  
            if (Math.Abs(_priorities[_maxPriorityIndex] - _maxPriority) == 0.0f) return;

            _maxPriority = 0.0f;
            _prioritiesSum = 0.0f;
            for (int i = 0; i < _experiences.Count; i++)
            {
                var priority = _priorities[i];
                _prioritiesSum += priority;

                if (_maxPriority > priority) continue;

                _maxPriority = priority;
                _maxPriorityIndex = i;
            }
        }
    }
}