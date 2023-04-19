using System.Collections.Generic;
using System.Diagnostics;
using NN;
using Unity.Burst;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;
using Unity.Jobs;
using Debug = UnityEngine.Debug;
using Random = UnityEngine.Random;

namespace Algorithms.RL
{
    public struct DuellingNetwork
    {
        public readonly NetworkModel InputLayer;
        public readonly NetworkModel ValueModel;
        public readonly NetworkModel AdvantageModel;

        public DuellingNetwork(NetworkModel inputLayer, NetworkModel valueModel, NetworkModel advantageModel)
        {
            InputLayer = inputLayer;
            ValueModel = valueModel;
            AdvantageModel = advantageModel;
        }
    }

    public struct ExperienceJobContainer
    {
        public int Action;
        public bool Done;
        public float Reward;

        public ExperienceJobContainer(int action, bool done, float reward)
        {
            Action = action;
            Done = done;
            Reward = reward;
        }
    }

    public class DuellingDQN : ModelDQN
    {
        private readonly DuellingNetwork _duellingNetwork;
        private readonly DuellingNetwork _duellingTarget;

        private readonly float[,] _actionSample;

        private readonly float[,] _dValue;
        private readonly float[,] _dAdvantage;
        private readonly float[,] _dInput;

        private readonly float _duellingDerivative;

        private readonly List<long> _times;

        private NativeArray<ExperienceJobContainer> __experiences;
        private NativeArray<float> _valueTarget;
        private NativeArray<float> _advantageTarget;
        private NativeArray<float> _valuePredict;
        private NativeArray<float> _advantagePredict;
        private NativeArray<float> __yTarget;
        private NativeArray<float> _qPredict;
        private NativeArray<float> __dAdvantage;
        private NativeArray<float> __dValue;
        private DuellingJob _duellingJob;

        public DuellingDQN(DuellingNetwork network, DuellingNetwork target, int numberOfActions, int stateSize,
            int maxExperienceSize = 10000, int minExperienceSize = 100, int batchSize = 32, float gamma = 0.99f) : base(
            null, null, numberOfActions, stateSize, maxExperienceSize, minExperienceSize, batchSize,
            gamma)
        {
            _duellingNetwork = network;
            _duellingTarget = target;

            _actionSample = new float[batchSize, numberOfActions];

            _dValue = new float[batchSize, 1];
            _dAdvantage = new float[batchSize, numberOfActions];

            // TODO: do not use magic number
            _dInput = new float[batchSize, 128];

            _duellingDerivative = 1.0f - 1.0f / numberOfActions;

            _times = new List<long>();

            __experiences = new NativeArray<ExperienceJobContainer>(batchSize, Allocator.Persistent);
            _valueTarget = new NativeArray<float>(batchSize, Allocator.Persistent);
            _advantageTarget = new NativeArray<float>(batchSize * numberOfActions, Allocator.Persistent);
            _valuePredict = new NativeArray<float>(batchSize, Allocator.Persistent);
            _advantagePredict = new NativeArray<float>(batchSize * numberOfActions, Allocator.Persistent);
            __yTarget = new NativeArray<float>(batchSize * numberOfActions, Allocator.Persistent);
            _qPredict = new NativeArray<float>(batchSize * numberOfActions, Allocator.Persistent);
            __dAdvantage = new NativeArray<float>(batchSize * numberOfActions, Allocator.Persistent);
            __dValue = new NativeArray<float>(batchSize, Allocator.Persistent);

            _duellingJob = new DuellingJob()
            {
                Experiences = __experiences,
                ValueTarget = _valueTarget,
                AdvantageTarget = _advantageTarget,
                ValuePredict = _valuePredict,
                AdvantagePredict = _advantagePredict,

                YTarget = __yTarget,
                QPredict = _qPredict,

                DAdvantage = __dAdvantage,
                DValue = __dValue,

                BatchSize = batchSize,
                NumberOfActions = numberOfActions,
                Gamma = _gamma,
                DuellingDerivative = _duellingDerivative
            };
        }

        public override int EpsilonGreedySample(float[] state, float eps = 0.1f)
        {
            var probability = Random.value;
            if (probability > eps)
            {
                for (int i = 0; i < _stateLenght; i++)
                {
                    _predictSate[0, i] = state[i];
                }

                MaxByRow(_duellingNetwork.AdvantageModel.Predict(_duellingNetwork.InputLayer.Predict(_predictSate)),
                    true);
                return _nextQ[0].index;
            }

            return Random.Range(0, _numberOfActions);
        }

        public override void Train()
        {
            if (_experiences.Count < _minExperienceSize) return;

            RandomBatch();

            var inputTarget = _duellingTarget.InputLayer.Predict(_nextStates);
            var valueTarget = _duellingTarget.ValueModel.Predict(inputTarget);
            var advantageTarget = _duellingTarget.AdvantageModel.Predict(inputTarget);

            var inputPredict = _duellingNetwork.InputLayer.Predict(_currentStates);
            var valuePredict = _duellingNetwork.ValueModel.Predict(inputPredict);
            var advantagePredict = _duellingNetwork.AdvantageModel.Predict(inputPredict);

            var stopwatch = new Stopwatch();
            stopwatch.Start();

            for (int i = 0; i < _batchSize; i++)
            {
                var experience = _experiences[_batchIndexes[i]];
                __experiences[i] = new ExperienceJobContainer(experience.Action, experience.Done, experience.Reward);
                _valueTarget[i] = valueTarget[i, 0];
                _valuePredict[i] = valuePredict[i, 0];

                var currentRow = i * _numberOfActions;
                for (int j = 0; j < _numberOfActions; j++)
                {
                    _advantageTarget[currentRow + j] = advantageTarget[i, j];
                    _advantagePredict[currentRow + j] = advantagePredict[i, j];
                }
            }

            var duellingJobHandle = _duellingJob.Schedule(_batchSize, 32);
            duellingJobHandle.Complete();

            for (int i = 0; i < _batchSize; i++)
            {
                _dValue[i, 0] = __dValue[i];
                var currentRow = i * _numberOfActions;
                for (int j = 0; j < _numberOfActions; j++)
                {
                    _dAdvantage[i, j] = __dAdvantage[currentRow + j];
                }
            }

            //TODO: Could possible be done in a job
            // for (int i = 0; i < _batchSize; i++)
            // {
            //     var predictAdvantageMean = 0.0f;
            //     for (int j = 0; j < _numberOfActions; j++)
            //     {
            //         predictAdvantageMean += advantagePredict[i, j];
            //     }
            //
            //     predictAdvantageMean /= _numberOfActions;
            //
            //     var stateValue = valuePredict[i, 0];
            //     var targetAdvantageMean = 0.0f;
            //     var maxTargetAdvantage = advantageTarget[i, 0];
            //
            //     for (int j = 0; j < _numberOfActions; j++)
            //     {
            //         var qValuePredict = stateValue + (advantagePredict[i, j] - predictAdvantageMean);
            //         _actionSample[i, j] = qValuePredict;
            //         _yTarget[i, j] = qValuePredict;
            //
            //         var currentTargetAdvantage = advantageTarget[i, j];
            //         targetAdvantageMean += currentTargetAdvantage;
            //
            //         if (maxTargetAdvantage > currentTargetAdvantage) continue;
            //
            //         maxTargetAdvantage = currentTargetAdvantage;
            //     }
            //
            //     var experience = _experiences[_batchIndexes[i]];
            //     targetAdvantageMean /= _numberOfActions;
            //     var qValueTarget = valueTarget[i, 0] + (maxTargetAdvantage - targetAdvantageMean);
            //     _yTarget[i, experience.Action] =
            //         experience.Done ? experience.Reward : experience.Reward + _gamma * qValueTarget;
            //
            //     var dValue = 0.0f;
            //     //for (int k = 0; k < _numberOfActions; k++) _dAdvantage[i, k] = 0.0f;
            //     for (int j = 0; j < _numberOfActions; j++)
            //     {
            //         var dAdvantage = 0.0f;
            //         if (experience.Action == j)
            //         {
            //             //TODO: dividing by the number of batches might no longer make sense because we only updating one value 
            //             dValue = -2.0f * (_yTarget[i, j] - _actionSample[i, j]) / _numberOfActions / _batchSize;
            //             dAdvantage = dValue * _duellingDerivative;
            //         }
            //
            //         _dAdvantage[i, j] = dAdvantage;
            //
            //         // TODO: Derivative according to: https://datascience.stackexchange.com/questions/54023/dueling-network-gradient-with-respect-to-advantage-stream
            //         // float loss = -2.0f * (_yTarget[i, j] - _actionSample[i, j]) / _numberOfActions / _batchSize;
            //         // dValue += loss;
            //         // for (int k = 0; k < _numberOfActions; k++)
            //         // {
            //         //     _dAdvantage[i, k] += j == k ? loss * _duellingDerivative : loss * -0.1f;
            //         // }
            //     }
            //
            //     _dValue[i, 0] = dValue;
            // }

            stopwatch.Stop();
            _times.Add(stopwatch.ElapsedTicks);

            var dValueFinal = _duellingNetwork.ValueModel.Update(_dValue);
            var dAdvantageFinal = _duellingNetwork.AdvantageModel.Update(_dAdvantage);
            for (int i = 0; i < _batchSize; i++)
            {
                for (int j = 0; j < _dInput.GetLength(1); j++)
                {
                    _dInput[i, j] = dValueFinal[i, j] + dAdvantageFinal[i, j];
                }
            }

            _duellingNetwork.InputLayer.Update(_dInput);
        }

        public override float SampleLoss()
        {
            // Because all variables have been initialized in the train function there is no need to do it here
            // basically this gives the loss for the previous batch 
            if (_experiences.Count < _minExperienceSize) return 0.0f;

            float loss = 0;
            for (int i = 0; i < _batchSize; i++)
            {
                var result = 0.0f;
                for (int j = 0; j < _numberOfActions; j++)
                {
                    float error = _yTarget[i, j] - _actionSample[i, j];
                    result += error * error;
                }

                loss += result / _numberOfActions;
            }

            loss /= _batchSize;
            return loss;
        }

        public override void SetTargetModel()
        {
            _duellingNetwork.InputLayer.CopyModel(_duellingTarget.InputLayer);
            _duellingNetwork.ValueModel.CopyModel(_duellingTarget.ValueModel);
            _duellingNetwork.AdvantageModel.CopyModel(_duellingTarget.AdvantageModel);
        }

        public override void Dispose()
        {
            _duellingNetwork.InputLayer.Dispose();
            _duellingNetwork.ValueModel.Dispose();
            _duellingNetwork.AdvantageModel.Dispose();

            _duellingTarget.InputLayer.Dispose();
            _duellingTarget.ValueModel.Dispose();
            _duellingTarget.AdvantageModel.Dispose();

            // job
            __experiences.Dispose();
            _valueTarget.Dispose();
            _advantageTarget.Dispose();
            _valuePredict.Dispose();
            _advantagePredict.Dispose();
            __yTarget.Dispose();
            _qPredict.Dispose();
            __dAdvantage.Dispose();
            __dValue.Dispose();

            var average = 0.0f;
            foreach (var t in _times)
                average += t;

            Debug.Log("Time: " + (average / _times.Count));
        }

        [BurstCompile]
        private struct DuellingJob : IJobParallelFor
        {
            [ReadOnly] public NativeArray<ExperienceJobContainer> Experiences;
            [ReadOnly] public NativeArray<float> ValueTarget;
            [ReadOnly] public NativeArray<float> AdvantageTarget;
            [ReadOnly] public NativeArray<float> ValuePredict;
            [ReadOnly] public NativeArray<float> AdvantagePredict;

            [NativeDisableContainerSafetyRestriction] public NativeArray<float> YTarget;
            [NativeDisableContainerSafetyRestriction] public NativeArray<float> QPredict;

            [NativeDisableContainerSafetyRestriction] public NativeArray<float> DAdvantage;
            [WriteOnly] public NativeArray<float> DValue;

            public int BatchSize;
            public int NumberOfActions;
            public float Gamma;
            public float DuellingDerivative;

            public void Execute(int index)
            {
                var currentRow = index * NumberOfActions;

                var predictAdvantageMean = 0.0f;
                for (int i = 0; i < NumberOfActions; i++)
                {
                    predictAdvantageMean += AdvantagePredict[currentRow + i];
                }

                predictAdvantageMean /= NumberOfActions;

                var stateValue = ValuePredict[index];
                var targetAdvantageMean = 0.0f;
                var maxTargetAdvantage = AdvantageTarget[currentRow];

                for (int i = 0; i < NumberOfActions; i++)
                {
                    var qValuePredict = stateValue + (AdvantagePredict[currentRow + i] - predictAdvantageMean);
                    QPredict[currentRow + i] = qValuePredict;
                    YTarget[currentRow + i] = qValuePredict;

                    var currentTargetAdvantage = AdvantageTarget[currentRow + i];
                    targetAdvantageMean += currentTargetAdvantage;

                    if (maxTargetAdvantage > currentTargetAdvantage) continue;

                    maxTargetAdvantage = currentTargetAdvantage;
                }

                var experience = Experiences[index];
                targetAdvantageMean /= NumberOfActions;
                var qValueTarget = ValueTarget[index] + (maxTargetAdvantage - targetAdvantageMean);
                YTarget[currentRow + experience.Action] =
                    experience.Done ? experience.Reward : experience.Reward + Gamma * qValueTarget;

                var dValue = 0.0f;
                for (int i = 0; i < NumberOfActions; i++)
                {
                    var dAdvantage = 0.0f;
                    if (experience.Action == i)
                    {
                        dValue = -2.0f * (YTarget[currentRow + i] - QPredict[currentRow + i]) / NumberOfActions /
                                 BatchSize;
                        dAdvantage = dValue * DuellingDerivative;
                    }

                    DAdvantage[currentRow + i] = dAdvantage;
                }

                DValue[index] = dValue;
            }
        }
    }
}

// for (int i = 0; i < _batchSize; i++)
// {
//     for (int j = 0; j < _numberOfActions; j++)
//     {
//         float dAdvantage = 0.0f;
//         for (int k = 0; k < _numberOfActions; k++)
//         {
//             dAdvantage += j == k ?  _loss[i, k] * _duellingDerivative :  _loss[i, k] * (1.0f / _numberOfActions);
//         }
//         _dAdvantage[i,j] = dAdvantage;
//
//     }
// }