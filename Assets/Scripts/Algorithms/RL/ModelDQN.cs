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
            public readonly float[] CurrentState;
            public readonly int Action;
            public readonly float Reward;
            public readonly bool Done;
            public readonly float[] NextState;

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
        private readonly int _numberOfActions;
        private readonly int _maxExperienceSize;
        protected readonly int _minExperienceSize;
        protected readonly int _batchSize;
        protected readonly float _gamma;

        protected readonly List<Experience> _experiences;

        // cached variables
        protected readonly float[,] _nextStates;
        protected readonly float[,] _currentStates;
        protected readonly int[] _batchIndexes;
        protected readonly (int index, float value)[] _nextQ;
        protected readonly float[,] _yTarget;
        private readonly float[,] _predictSate;

        public ModelDQN(NetworkModel networkModel, NetworkModel targetModel, int numberOfActions, int stateSize,
            int maxExperienceSize = 10000, int minExperienceSize = 100, int batchSize = 32, float gamma = 0.99f)
        {
            _networkModel = networkModel;
            _targetModel = targetModel;
            _numberOfActions = numberOfActions;
            _maxExperienceSize = maxExperienceSize;
            _minExperienceSize = minExperienceSize;
            _batchSize = batchSize;
            _gamma = gamma;

            _experiences = new List<Experience>(maxExperienceSize);

            _nextStates = new float[batchSize, stateSize];
            _currentStates = new float[batchSize, stateSize];
            _batchIndexes = new int[batchSize];
            _nextQ = new (int index, float value)[batchSize];
            _yTarget = new float[batchSize, numberOfActions];
            _predictSate = new float[1, stateSize];
        }

        public int EpsilonGreedySample(float[] state, float eps = 0.1f)
        {
            var probability = Random.value;
            if (probability > eps)
            {
                for (int i = 0; i < state.Length; i++)
                {
                    _predictSate[0, i] = state[i];
                }
                
                MaxByRow( _networkModel.Predict(_predictSate));
                return _nextQ[0].index;
            }

            return Random.Range(0, _numberOfActions);
        }

        public void AddExperience(float[] currentState, int action, float reward, bool done, float[] nextState)
        {
            var experience = new Experience(currentState, action, reward, done, nextState);
            _experiences.Add(experience);
            if (_experiences.Count >= _maxExperienceSize)
            {
                _experiences.RemoveAt(0);
            }
        }

        public virtual void Train()
        {
            if (_experiences.Count < _minExperienceSize)
                return;

            RandomBatch();

            MaxByRow(_targetModel.Predict(_nextStates));
            NnMath.CopyMatrix(_yTarget, _networkModel.Predict(_currentStates));
            for (int i = 0; i < _nextQ.Length; i++)
            {
                var experience = _experiences[_batchIndexes[i]];
                _yTarget[i, experience.Action] =
                    experience.Done ? experience.Reward : experience.Reward + _gamma * _nextQ[i].value;
            }

            _networkModel.Update(_yTarget);
        }

        public float SampleLoss()
        {
            // Because all variables have been initialized in the train function there is no need to do it here
            // basically this gives the loss for the previous batch 
            return _networkModel.Loss(_yTarget);
        }

        public void SetTargetModel()
        {
            _networkModel.CopyModel(_targetModel);
        }

        public void Dispose()
        {
            _networkModel.Dispose();
            _targetModel.Dispose();
        }

        protected virtual void RandomBatch()
        {
            for (int i = 0; i < _batchSize; i++)
            {
                _batchIndexes[i] = -1;
            }

            var iterations = 0;
            do
            {
                var index = Random.Range(0, _experiences.Count);
                var hasIndex = false;

                for (int i = 0; i < iterations + 1; i++)
                {
                    if (_batchIndexes[i] != index) continue;

                    hasIndex = true;
                    break;
                }

                if (hasIndex) continue;

                _batchIndexes[iterations] = index;
                
                var experience = _experiences[index];
                for (int i = 0; i < experience.CurrentState.Length; i++)
                {
                    _nextStates[iterations, i] = experience.NextState[i];
                    _currentStates[iterations, i] = experience.CurrentState[i];
                }

                ++iterations;
            } while (iterations < _batchSize);
        }

        protected void MaxByRow(float[,] matrix, bool firstRow = false)
        {
            int sampleSize = firstRow ? 1 : matrix.GetLength(0);
            for (int i = 0; i < sampleSize; i++)
            {
                int maxIndex = 0;
                float maxValue = float.MinValue;
                for (int j = 0; j < matrix.GetLength(1); j++)
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