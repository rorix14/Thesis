using System;
using NN.CPU_Single;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Algorithms.RL
{
    public class RainbowDQN : ModelDQN
    {
        //Dist
        private readonly int _supportSize;
        private readonly float[] _support;
        private readonly float _vMin;
        private readonly float _vMax;
        private readonly float _supportDelta;

        //Duelling 
        private readonly DuellingNetwork _duellingNetwork;
        private readonly DuellingNetwork _duellingTarget;
        private readonly float[,] _actionSample;
        private readonly float _duellingDerivative;
        private readonly float[,] _dValue;
        private readonly float[,] _dAdvantage;
        private readonly float[,] _dInput;

        private float[,] _inputTarget;
        private float[,] _valueTarget;
        private float[,] _advantageTarget;
        private float[,] _inputPredict;
        private float[,] _valuePredict;
        private float[,] _advantagePredict;

        //Double & Duelling
        private float[,] _inputNextPredict;
        private float[,] _valueMextPredict;
        private float[,] _advantageNextPredict;

        //N-Step
        private readonly int _nStep;
        private readonly float[] _storedNStepGammas;
        private readonly Experience[] _nStepBuffer;
        private int _lastNStepPosition;

        //PER
        private readonly float _alpha;
        private readonly float _initialBeta;
        private float _beta;
        private readonly float[] _sampleWeights;
        private int _counter;
        private float _maxPriority;
        private int _maxPriorityIndex;
        private float _maxWeight;
        private readonly SumTree _sumTree;
        private readonly float[] _sampleLosses;

        //Cashed variables 
        private readonly int _inputNumberOfNeurons;
        private readonly float[] _predictNextAdvantageMean;
        private readonly float[] _predictAdvantageMean;
        private readonly float[] _targetAdvantageMean;
        private readonly float[] _qTargetDistribution;
        private readonly float[] _unProjectedTargetDist;
        private float[,] _dValueFinal;
        private float[,] _dAdvantageFinal;

        public RainbowDQN(DuellingNetwork network, DuellingNetwork target, int numberOfActions, int stateSize,
            int nStep, int supportSize, float vMin, float vMax, float beta = 0.4f, float alpha = 0.6f,
            int maxExperienceSize = 10000, int minExperienceSize = 100, int batchSize = 32, float gamma = 0.99f) : base(
            null, null, numberOfActions, stateSize, maxExperienceSize, minExperienceSize, batchSize,
            gamma)
        {
            _yTarget = new float[batchSize, supportSize];

            //Dist
            _supportSize = supportSize;
            _vMin = vMin;
            _vMax = vMax;
            _supportDelta = (vMax - vMin) / (supportSize - 1);

            _support = new float[supportSize];
            for (int i = 0; i < supportSize; i++)
            {
                _support[i] = vMin + _supportDelta * i;
            }

            //Duelling 
            _duellingNetwork = network;
            _duellingTarget = target;
            _actionSample = new float[batchSize, supportSize];
            _duellingDerivative = 1.0f - 1.0f / numberOfActions;
            _dValue = new float[batchSize, supportSize];
            _dAdvantage = new float[batchSize, numberOfActions * supportSize];
            //TODO: Make this not a magic number, can have model class have a shape variable where it has all the sizes
            _inputNumberOfNeurons = 128;
            _dInput = new float[batchSize, _inputNumberOfNeurons];

            //N-Step
            _nStep = nStep <= 0 ? 1 : nStep;
            _nStepBuffer = new Experience[nStep];
            _storedNStepGammas = new float[_nStep + 1];
            for (int i = 0; i < _nStep + 1; i++)
            {
                _storedNStepGammas[i] = Mathf.Pow(_gamma, i);
            }

            //PER
            _alpha = alpha;
            _beta = beta;
            _initialBeta = beta;
            _sampleWeights = new float[batchSize];
            _counter = 0;
            _maxPriority = 1.0f;
            _maxPriorityIndex = 0;
            _sumTree = new SumTree(maxExperienceSize);
            _sampleLosses = new float[batchSize];

            //Noisy 
            _predictSate = new float[batchSize, stateSize];

            //Cashed
            _predictNextAdvantageMean = new float[supportSize];
            _predictAdvantageMean = new float[supportSize];
            _targetAdvantageMean = new float[supportSize];
            _unProjectedTargetDist = new float[supportSize];
            _qTargetDistribution = new float[supportSize];
        }

        public override int EpsilonGreedySample(float[] state, float eps = 0.1f)
        {
            //if (!(Random.value > eps)) return Random.Range(0, _numberOfActions);
            for (int i = 0; i < _stateLenght; i++)
            {
                _predictSate[0, i] = state[i];
            }

            _inputPredict = _duellingNetwork.InputLayer.Predict(_predictSate);
            _valuePredict = _duellingNetwork.ValueModel.Predict(_inputPredict);
            _advantagePredict = _duellingNetwork.AdvantageModel.Predict(_inputPredict);

            for (int i = 0; i < _supportSize; i++)
            {
                _predictAdvantageMean[i] = 0.0f;
                for (int j = 0; j < _numberOfActions; j++)
                {
                    _predictAdvantageMean[i] += _advantagePredict[0, j * _supportSize + i];
                }

                _predictAdvantageMean[i] /= _numberOfActions;
            }

            var maxQValue = float.MinValue;
            var mMaxQIndex = 0;
            for (int i = 0; i < _numberOfActions; i++)
            {
                var maxValue = float.MinValue;
                var startIndex = _supportSize * i;
                for (int j = 0; j < _supportSize; j++)
                {
                    var value = _valuePredict[0, j] + (_advantagePredict[0, startIndex + j] - _predictAdvantageMean[j]);
                    _qTargetDistribution[j] = value;

                    if (maxValue > value) continue;
                    maxValue = value;
                }

                var expSum = 0.0f;
                for (int j = 0; j < _supportSize; j++)
                {
                    var result = Mathf.Exp(_qTargetDistribution[j] - maxValue);
                    _qTargetDistribution[j] = result;
                    expSum += result;
                }

                var qValue = 0f;
                for (int j = 0; j < _supportSize; j++)
                {
                    qValue += _qTargetDistribution[j] / expSum * _support[j];
                }

                if (maxQValue > qValue) continue;

                maxQValue = qValue;
                mMaxQIndex = i;
            }

            return mMaxQIndex;
        }

        public override void AddExperience(float[] currentState, int action, float reward, bool done, float[] nextState)
        {
            _nStepBuffer[_lastNStepPosition] = new Experience(currentState, action, reward, done, nextState);
            _lastNStepPosition = (_lastNStepPosition + 1) % _nStep;

            var experience = _nStepBuffer[_lastNStepPosition];
            if (experience.CurrentState == null) return;

            if (!experience.Done)
            {
                for (int i = 1; i < _nStep; i++)
                {
                    var nStepExperience = _nStepBuffer[(_lastNStepPosition + i) % _nStep];
                    experience.Done = nStepExperience.Done;
                    experience.NextState = nStepExperience.NextState;
                    experience.Reward += _storedNStepGammas[i] * nStepExperience.Reward;

                    if (nStepExperience.Done) break;
                }
            }

            if (_experiences.Count < _maxExperienceSize)
            {
                _experiences.Add(experience);
            }
            else
            {
                _experiences[_lastExperiencePosition] = experience;
            }

            _sumTree.UpdateValue(_lastExperiencePosition, _maxPriority);
            _lastExperiencePosition = (_lastExperiencePosition + 1) % _maxExperienceSize;
        }


        public override void Train()
        {
            if (_experiences.Count < _minExperienceSize)
                return;

            UpdateBeta();
            RandomBatch();

            _inputNextPredict = _duellingNetwork.InputLayer.Predict(_nextStates);
            _valueMextPredict = _duellingNetwork.ValueModel.Predict(_inputNextPredict);
            _advantageNextPredict = _duellingNetwork.AdvantageModel.Predict(_inputNextPredict);

            _inputTarget = _duellingTarget.InputLayer.Predict(_nextStates);
            _valueTarget = _duellingTarget.ValueModel.Predict(_inputTarget);
            _advantageTarget = _duellingTarget.AdvantageModel.Predict(_inputTarget);

            _inputPredict = _duellingNetwork.InputLayer.Predict(_currentStates);
            _valuePredict = _duellingNetwork.ValueModel.Predict(_inputPredict);
            _advantagePredict = _duellingNetwork.AdvantageModel.Predict(_inputPredict);

            for (int i = 0; i < _batchSize; i++)
            {
                int actionIndex;
                for (int j = 0; j < _supportSize; j++)
                {
                    _yTarget[i, j] = 0.0f;
                    _predictNextAdvantageMean[j] = 0.0f;
                    _predictAdvantageMean[j] = 0.0f;
                    _targetAdvantageMean[j] = 0.0f;

                    for (int k = 0; k < _numberOfActions; k++)
                    {
                        actionIndex = k * _supportSize + j;

                        _dAdvantage[i, actionIndex] = 0.0f;

                        _predictNextAdvantageMean[j] += _advantageNextPredict[i, actionIndex];
                        _predictAdvantageMean[j] += _advantagePredict[i, actionIndex];
                        _targetAdvantageMean[j] += _advantageTarget[i, actionIndex];
                    }

                    _predictNextAdvantageMean[j] /= _numberOfActions;
                    _predictAdvantageMean[j] /= _numberOfActions;
                    _targetAdvantageMean[j] /= _numberOfActions;
                }

                float maxValue;
                var targetMaxQValue = float.MinValue;
                var targetMaxQIndex = 0;
                for (int j = 0; j < _numberOfActions; j++)
                {
                    maxValue = float.MinValue;
                    actionIndex = _supportSize * j;
                    for (int k = 0; k < _supportSize; k++)
                    {
                        var value = _valueMextPredict[i, k] +
                                    (_advantageNextPredict[i, actionIndex + k] - _predictNextAdvantageMean[k]);
                        _qTargetDistribution[k] = value;

                        if (maxValue > value) continue;
                        maxValue = value;
                    }

                    var targetExpSum = 0.0f;
                    for (int k = 0; k < _supportSize; k++)
                    {
                        var result = Mathf.Exp(_qTargetDistribution[k] - maxValue);
                        _qTargetDistribution[k] = result;
                        targetExpSum += result;
                    }

                    var qValue = 0f;
                    for (int k = 0; k < _supportSize; k++)
                    {
                        qValue += _qTargetDistribution[k] / targetExpSum * _support[k];
                    }

                    if (targetMaxQValue > qValue) continue;

                    targetMaxQValue = qValue;
                    targetMaxQIndex = j;
                }

                var experience = _experiences[_batchIndexes[i]];
                var experienceAction = experience.Action;
                maxValue = float.MinValue;
                actionIndex = experienceAction * _supportSize;
                var distActionIndex = targetMaxQIndex * _supportSize;
                var distMaxValue = float.MinValue;
                for (int j = 0; j < _supportSize; j++)
                {
                    var value = _valuePredict[i, j] +
                                (_advantagePredict[i, actionIndex + j] - _predictAdvantageMean[j]);
                    _actionSample[i, j] = value;
                    if (maxValue < value)
                    {
                        maxValue = value;
                    }

                    value = _valueTarget[i, j] + (_advantageTarget[i, distActionIndex + j] - _targetAdvantageMean[j]);
                    _unProjectedTargetDist[j] = value;
                    if (distMaxValue > value) continue;
                    distMaxValue = value;
                }

                var predictExpSum = 0.0f;
                var expSum = 0.0f;
                for (int j = 0; j < _supportSize; j++)
                {
                    var result = Mathf.Exp(_actionSample[i, j] - maxValue);
                    _actionSample[i, j] = result;
                    predictExpSum += result;

                    result = Mathf.Exp(_unProjectedTargetDist[j] - distMaxValue);
                    _unProjectedTargetDist[j] = result;
                    expSum += result;
                }

                for (int j = 0; j < _supportSize; j++)
                {
                    var value = experience.Done
                        ? experience.Reward
                        : experience.Reward + _support[j] * _storedNStepGammas[_nStep];
                    var tz = NnMath.Clamp(value, _vMin, _vMax);
                    var b = (tz - _vMin) / _supportDelta;
                    var lower = (int)b;
                    var upper = Mathf.CeilToInt(b);

                    var probability = _unProjectedTargetDist[j] / expSum;
                    if (lower == upper)
                    {
                        _yTarget[i, lower] += probability;
                    }
                    else
                    {
                        _yTarget[i, lower] += probability * (upper - b);
                        _yTarget[i, upper] += probability * (b - lower);
                    }
                }

                //Loss and Derivatives
                actionIndex = experienceAction * _supportSize;
                var sampleLoss = 0.0f;
                var weightPer = _sampleWeights[i] / _maxWeight;
                for (int j = 0; j < _supportSize; j++)
                {
                    var predValue = _actionSample[i, j] / predictExpSum;
                    var targetValue = _yTarget[i, j];

                    var outputValue = NnMath.Clamp(predValue, 1e-7f, 1f - 1e-7f);
                    sampleLoss += targetValue * Mathf.Log(outputValue);

                    var dValue = (predValue - targetValue) * weightPer / _batchSize;

                    //TODO: need to make sure derivative of the advantage is correct
                    _dAdvantage[i, actionIndex + j] = dValue * _duellingDerivative;
                    _dValue[i, j] = dValue;
                }

                _sampleLosses[i] = sampleLoss / _supportSize * -1f;
            }

            _dValueFinal = _duellingNetwork.ValueModel.Update(_dValue);
            _dAdvantageFinal = _duellingNetwork.AdvantageModel.Update(_dAdvantage);
            for (int i = 0; i < _batchSize; i++)
            {
                for (int j = 0; j < _inputNumberOfNeurons; j++)
                {
                    _dInput[i, j] = _dValueFinal[i, j] + _dAdvantageFinal[i, j];
                }
            }

            _duellingNetwork.InputLayer.Update(_dInput);
            UpdatePriorities(_sampleLosses);
        }

        public override float SampleLoss()
        {
            if (_experiences.Count < _minExperienceSize) return 0.0f;

            float loss = 0;
            for (int i = 0; i < _batchSize; i++)
            {
                var result = 0.0f;
                for (int j = 0; j < _supportSize; j++)
                {
                    var outputValue = NnMath.Clamp(_actionSample[i, j], 1e-7f, 1f - 1e-7f);
                    result += _yTarget[i, j] * Mathf.Log(outputValue);
                }

                loss += result / _supportSize * -1f;
            }

            loss /= _batchSize;
            return loss;
        }

        protected override void RandomBatch()
        {
            var totalExperiences = _experiences.Count;

            var total = _sumTree.Total();
            _maxWeight = 0.0f;
            for (int i = 0; i < _batchSize; i++)
            {
                var rand = Random.Range(0.0f, total);
                var batchIndex = _sumTree.Sample(rand, out var priority);

                if (batchIndex >= totalExperiences)
                {
                    batchIndex = (_maxExperienceSize + _lastExperiencePosition - 1) % _maxExperienceSize;
                    priority = _sumTree.Get(batchIndex);
                }

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
                var priority = Mathf.Pow(samplePriorities[i] + 1e-5f, _alpha);
                var priorityIndex = _batchIndexes[i];

                _sumTree.UpdateValue(priorityIndex, priority);

                if (_maxPriority >= priority) continue;

                _maxPriority = priority;
                _maxPriorityIndex = priorityIndex;
            }

            if (Math.Abs(_sumTree.Get(_maxPriorityIndex) - _maxPriority) == 0.0f) return;

            var totalExperiences = _experiences.Count;
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

        // TODO: can probably remove this function
        protected override void MaxByRow(float[,] matrix, bool firstRow = false)
        {
            var sampleSize = firstRow ? 1 : _batchSize;
            for (int i = 0; i < sampleSize; i++)
            {
                var maxValue = float.MinValue;
                var maxIndex = 0;
                for (int j = 0; j < _numberOfActions; j++)
                {
                    var startIndex = j * _supportSize;
                    var qValue = 0f;
                    for (int k = 0; k < _supportSize; k++)
                    {
                        qValue += matrix[i, startIndex + k] * _support[k];
                    }

                    if (maxValue > qValue) continue;

                    maxValue = qValue;
                    maxIndex = j;
                }

                _nextQ[i] = (maxIndex, maxValue);
            }
        }

        public override void SetTargetModel()
        {
            _duellingNetwork.CopyModel(_duellingTarget);
        }

        public override void Dispose()
        {
            _duellingNetwork.Dispose();
            _duellingTarget.Dispose();
        }
    }
}