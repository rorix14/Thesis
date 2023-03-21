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
        private readonly float[] _samplePriorities;
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
            _samplePriorities = new float[batchSize];
            _counter = 0;
        }

        public override void AddExperience(float[] currentState, int action, float reward, bool done, float[] nextState)
        {
            var experience = new Experience(currentState, action, reward, done, nextState);
            float maxPriority = NnMath.ArrayMax(_priorities);

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
            _counter++;
        }

        public override void Train()
        {
            if (_experiences.Count < _minExperienceSize)
                return;

            UpdateBeta();
            RandomBatch();

            MaxByRow(_targetModel.Predict(_nextStates));
            NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));
            for (int i = 0; i < _nextQ.Length; i++)
            {
                var experience = _experiences[_batchIndexes[i]];
                _yTarget[i, experience.Action] =
                    experience.Done ? experience.Reward : experience.Reward + _gamma * _nextQ[i].value;
            }

            _networkModel.SetLossParams(_sampleWeights);
            
            // TODO: This can just return the sample losses instead of the loss mean
            _networkModel.Loss(_yTarget);
            
            // TODO: compare speed of this function with a manual copy
            _networkModel._lossFunction.SampleLosses.CopyTo(_samplePriorities, 0);
            
            UpdatePriorities();
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

            
            for (int i = 0; i < totalExperiences; i++)
            {
                _probabilities[i] /= prioritiesSum;
            }

            prioritiesSum = 0;
            for (int i = 0; i < totalExperiences; i++)
            {
                prioritiesSum += _probabilities[i];
            }

            for (int i = 0; i < _batchSize; i++)
            {
                // index could be a random number within the range of 0 and experience size
                var index = -1;
                var cutoff = Random.value;
                while (cutoff > 0)
                {
                   
                    index++;
                    if(index < 0 || index >= _experiences.Count)
                        Debug.Log("Wrong index: " + index + ", in a max of: " + (_experiences.Count - 1) + ", probability sum: " + prioritiesSum);
                    cutoff -= _probabilities[index];
                }

                
                _batchIndexes[i] = index;
                var experience = _experiences[index];
                for (int j = 0; j < experience.CurrentState.Length; j++)
                {
                    _nextStates[i, j] = experience.NextState[i];
                    _currentStates[i, j] = experience.CurrentState[i];
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
            var value = _initialBeta + _counter * (1.0f - _initialBeta) / 100000;
            _beta = 1.0f < value ? 1.0f : value;
        }

        private void UpdatePriorities()
        {
            for (int i = 0; i < _batchSize; i++)
            {
                _priorities[_batchIndexes[i]] = _samplePriorities[i] + 1e-5f;
            }
        }
    }
}