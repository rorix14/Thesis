using System.Collections.Generic;
using System.Diagnostics;
using NN;
using NN.CPU_Single;
using UnityEngine;

namespace Algorithms.RL
{
    public class ModelPrioritizedDQN : ModelDQN
    {
        private readonly float _alpha;
        private readonly float _initialBeta;
        private float _beta;
        private readonly float[] _priorities;
        private readonly float[] _probabilities;
        private readonly float[] _sampleWeights;
        private int _counter;

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
            _probabilities = new float[maxExperienceSize];
            _sampleWeights = new float[batchSize];
            _counter = 0;

            //_alpha = 0f;
        }

        public override void AddExperience(float[] currentState, int action, float reward, bool done, float[] nextState)
        {
            var experience = new Experience(currentState, action, reward, done, nextState);
            var maxPriority = NnMath.ArrayMax(_priorities);

            if (_experiences.Count < _maxExperienceSize)
            {
                _experiences.Add(experience);
                maxPriority = maxPriority > 0 ? maxPriority : 1;
            }
            else
            {
                _experiences[_lastExperiencePosition] = experience;
            }

            _priorities[_lastExperiencePosition] = maxPriority;
            _lastExperiencePosition = (_lastExperiencePosition + 1) % _maxExperienceSize;
        }

        public override void Train()
        {
            if (_experiences.Count < _minExperienceSize)
                return;

            UpdateBeta();
            RandomBatch();

            MaxByRow(_networkModel.Predict(_nextStates));
            var targetPredictions = _targetModel.Predict(_nextStates);
            NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));
            for (int i = 0; i < _nextQ.Length; i++)
            {
                var experience = _experiences[_batchIndexes[i]];
                _yTarget[i, experience.Action] =
                    experience.Done
                        ? experience.Reward
                        : experience.Reward + _gamma * targetPredictions[i, _nextQ[i].index];
            }
            
            // MaxByRow(_targetModel.Predict(_nextStates));
            // NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));
            // for (int i = 0; i < _nextQ.Length; i++)
            // {
            //     var experience = _experiences[_batchIndexes[i]];
            //     _yTarget[i, experience.Action] =
            //         experience.Done ? experience.Reward : experience.Reward + _gamma * _nextQ[i].value;
            // }

            _networkModel.SetLossParams(_sampleWeights);
            UpdatePriorities(_networkModel.Loss(_yTarget));
            
            _networkModel.Update(_yTarget);
        }

        protected override void RandomBatch()
        {
            var totalExperiences = _experiences.Count;

            var prioritiesSum = 0.0f;
            for (int i = 0; i < totalExperiences; i++)
            {
                var priority = Mathf.Pow(_priorities[i], _alpha);
                _probabilities[i] = priority;
                prioritiesSum += priority;
            }

            var totalProbability = 0.0f;
            for (int i = 0; i < totalExperiences; i++)
            {
                _probabilities[i] /= prioritiesSum;
                totalProbability += _probabilities[i];
            }
            
            for (int i = 0; i < _batchSize; i++)
            {
                var batchIndex = -1;
                var cutoff = Random.Range(1e-20f, totalProbability - 1e-5f);
                while (cutoff > 0.0)
                {
                    batchIndex++;
                    cutoff -= _probabilities[batchIndex];
                }
                
                _batchIndexes[i] = batchIndex;
                var experience = _experiences[batchIndex];
                for (int j = 0; j < experience.CurrentState.Length; j++)
                {
                    _nextStates[i, j] = experience.NextState[j];
                    _currentStates[i, j] = experience.CurrentState[j];
                }
            }

            var maxWeight = float.MinValue;
            for (int i = 0; i < _batchSize; i++)
            {
                var weight = Mathf.Pow(totalExperiences * _probabilities[_batchIndexes[i]], -_beta);
                _sampleWeights[i] = weight;

                if (maxWeight > weight) continue;

                maxWeight = weight;
            }

            for (int i = 0; i < _batchSize; i++)
            {
                _sampleWeights[i] /= maxWeight;
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
                _priorities[_batchIndexes[i]] = samplePriorities[i] + 1e-5f;
            }
        }
    }
}