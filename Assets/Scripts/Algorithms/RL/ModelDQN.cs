using System.Collections.Generic;
using System.Diagnostics;
using NN;
using NN.CPU_Single;
using Debug = UnityEngine.Debug;
using Random = UnityEngine.Random;

namespace Algorithms.RL
{
    public class ModelDQN
    {
        protected struct Experience
        {
            public float[] CurrentState;
            public int Action;
            public float Reward;
            public bool Done;
            public float[] NextState;

            public Experience(float[] currentState, int action, float reward, bool done, float[] nextState)
            {
                CurrentState = currentState;
                Action = action;
                Reward = reward;
                Done = done;
                NextState = nextState;
            }
        }

        protected readonly NetworkModel _networkModel;
        protected readonly NetworkModel _targetModel;
        protected readonly int _numberOfActions;
        protected readonly int _stateLenght;
        protected readonly int _maxExperienceSize;
        protected readonly int _minExperienceSize;
        protected readonly int _batchSize;
        protected readonly float _gamma;

        protected readonly List<Experience> _experiences;
        protected int _lastExperiencePosition;

        // cached variables
        protected readonly float[,] _nextStates;
        protected readonly float[,] _currentStates;
        protected readonly int[] _batchIndexes;
        protected readonly (int index, float value)[] _nextQ;
        protected readonly float[,] _yTarget;
        protected float[,] _predictSate;

        public ModelDQN(NetworkModel networkModel, NetworkModel targetModel, int numberOfActions, int stateSize,
            int maxExperienceSize = 10000, int minExperienceSize = 100, int batchSize = 32, float gamma = 0.99f)
        {
            _networkModel = networkModel;
            _targetModel = targetModel;
            _numberOfActions = numberOfActions;
            _stateLenght = stateSize;
            _maxExperienceSize = maxExperienceSize;
            _minExperienceSize = minExperienceSize;
            _batchSize = batchSize;
            _gamma = gamma;

            _experiences = new List<Experience>(maxExperienceSize);
            _lastExperiencePosition = 0;

            _nextStates = new float[batchSize, stateSize];
            _currentStates = new float[batchSize, stateSize];
            _batchIndexes = new int[batchSize];
            _nextQ = new (int index, float value)[batchSize];
            _yTarget = new float[batchSize, numberOfActions];
            _predictSate = new float[1, stateSize];
        }

        public virtual int EpsilonGreedySample(float[] state, float eps = 0.1f)
        {
            var probability = Random.value;
            //if (_experiences.Count >= _minExperienceSize && probability > eps)
            if (probability > eps)
            {
                for (int i = 0; i < _stateLenght; i++)
                {
                    _predictSate[0, i] = state[i];
                }

                MaxByRow(_networkModel.Predict(_predictSate), true);
                return _nextQ[0].index;
            }

            return Random.Range(0, _numberOfActions);
        }

        public virtual void AddExperience(float[] currentState, int action, float reward, bool done, float[] nextState)
        {
            var experience = new Experience(currentState, action, reward, done, nextState);
            if (_experiences.Count < _maxExperienceSize)
            {
                _experiences.Add(experience);
            }
            else
            {
                _experiences[_lastExperiencePosition] = experience;
                _lastExperiencePosition = (_lastExperiencePosition + 1) % _maxExperienceSize;
            }
        }

        public virtual void Train()
        {
            if (_experiences.Count < _minExperienceSize) return;

            RandomBatch();

            MaxByRow(_targetModel.Predict(_nextStates));
            NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));
            for (int i = 0; i < _batchSize; i++)
            {
                var experience = _experiences[_batchIndexes[i]];
                _yTarget[i, experience.Action] =
                    experience.Done ? experience.Reward : experience.Reward + _gamma * _nextQ[i].value;
            }

            _networkModel.Update(_yTarget);
        }

        public virtual float SampleLoss()
        {
            // Because all variables have been initialized in the train function there is no need to do it here
            // basically this gives the loss for the previous batch 
            if (_experiences.Count < _minExperienceSize) return 0.0f;

            var sampleLosses = _networkModel.Loss(_yTarget);
            float loss = 0;
            foreach (var t in sampleLosses)
            {
                loss += t;
            }

            loss /= sampleLosses.Length;
            return loss;
        }

        public virtual void SetTargetModel()
        {
            _networkModel.CopyModel(_targetModel);
        }

        public virtual void Dispose()
        {
            _networkModel.Dispose();
            _targetModel.Dispose();
        }

        protected virtual void RandomBatch()
        {
            var iteration = 0;
            while (iteration < _batchSize)
            {
                _batchIndexes[iteration] = -1;
                var index = Random.Range(0, _experiences.Count);
                var hasIndex = false;

                for (int i = 0; i < iteration + 1; i++)
                {
                    if (_batchIndexes[i] != index) continue;

                    hasIndex = true;
                    break;
                }

                if (hasIndex) continue;

                _batchIndexes[iteration] = index;

                var experience = _experiences[index];
                for (int i = 0; i < _stateLenght; i++)
                {
                    _nextStates[iteration, i] = experience.NextState[i];
                    _currentStates[iteration, i] = experience.CurrentState[i];
                }

                ++iteration;
            }
        }

        protected void MaxByRow(float[,] matrix, bool firstRow = false)
        {
            int sampleSize = firstRow ? 1 : _batchSize;
            for (int i = 0; i < sampleSize; i++)
            {
                int maxIndex = 0;
                float maxValue = matrix[i, 0];
                for (int j = 1; j < _numberOfActions; j++)
                {
                    var currentVal = matrix[i, j];
                    if (maxValue > currentVal) continue;

                    maxIndex = j;
                    maxValue = currentVal;
                }

                _nextQ[i] = (maxIndex, maxValue);
            }
        }

        // Consider using jobs if batch size and action number is very large 
/*
        private struct IndexToValue
        {
            public int Index;
            public float Value;

            public IndexToValue(int index, float value)
            {
                Index = index;
                Value = value;
            }
        }

        [BurstCompile]
        private struct MaxByRowJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<float> Predictions;
            [WriteOnly] public NativeArray<float> NextQ;

            public int RowSize;

            public void Execute(int index)
            {
                int maxIndex = 0;
                var maxValue = float.MinValue;
                for (int i = 0; i < RowSize; i++)
                {
                    var currentVal = Predictions[index * RowSize + i];
                    if (maxValue > currentVal) continue;

                    maxIndex = i;
                    maxValue = currentVal;
                }

                NextQ[index] = maxValue;
            }
        }
        */
    }
}