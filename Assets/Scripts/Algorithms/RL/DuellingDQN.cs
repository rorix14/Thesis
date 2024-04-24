using DL.NN;
using NN;
using Random = UnityEngine.Random;

namespace Algorithms.RL
{
    public readonly struct DuellingNetwork
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

        public void CopyModel(DuellingNetwork otherModel)
        {
            InputLayer.CopyModel(otherModel.InputLayer);
            ValueModel.CopyModel(otherModel.ValueModel);
            AdvantageModel.CopyModel(otherModel.AdvantageModel);
        }

        public void Dispose()
        {
            InputLayer.Dispose();
            ValueModel.Dispose();
            AdvantageModel.Dispose();
        }
    }

    public class DuellingDQN : ModelDQN
    {
        private readonly DuellingNetwork _duellingNetwork;
        private readonly DuellingNetwork _duellingTarget;

        private readonly float[] _actionSample;

        private readonly float[,] _dValue;
        private readonly float[,] _dAdvantage;
        private readonly float[,] _dInput;

        private readonly float _duellingDerivative;

        // cashed outputs
        private float[,] _inputTarget;
        private float[,] _valueTarget;
        private float[,] _advantageTarget;
        private float[,] _inputPredict;
        private float[,] _valuePredict;
        private float[,] _advantagePredict;


        public DuellingDQN(DuellingNetwork network, DuellingNetwork target, int numberOfActions, int stateSize,
            int maxExperienceSize = 10000, int minExperienceSize = 100, int batchSize = 32, float gamma = 0.99f) : base(
            null, null, numberOfActions, stateSize, maxExperienceSize, minExperienceSize, batchSize,
            gamma)
        {
            _duellingNetwork = network;
            _duellingTarget = target;

            _actionSample = new float[batchSize];

            _dValue = new float[batchSize, 1];
            _dAdvantage = new float[batchSize, numberOfActions];

            // TODO: do not use magic number
            _dInput = new float[batchSize, 128];

            _duellingDerivative = 1.0f - 1.0f / numberOfActions;
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

            _inputTarget = _duellingTarget.InputLayer.Predict(_nextStates);
            _valueTarget = _duellingTarget.ValueModel.Predict(_inputTarget);
            _advantageTarget = _duellingTarget.AdvantageModel.Predict(_inputTarget);

            _inputPredict = _duellingNetwork.InputLayer.Predict(_currentStates);
            _valuePredict = _duellingNetwork.ValueModel.Predict(_inputPredict);
            _advantagePredict = _duellingNetwork.AdvantageModel.Predict(_inputPredict);

            for (int i = 0; i < _batchSize; i++)
            {
                var predictAdvantageMean = 0.0f;
                var targetAdvantageMean = 0.0f;
                var maxTargetAdvantage = _advantageTarget[i, 0];
                for (int j = 0; j < _numberOfActions; j++)
                {
                    _dAdvantage[i, j] = 0.0f;

                    predictAdvantageMean += _advantagePredict[i, j];

                    var currentTargetAdvantage = _advantageTarget[i, j];
                    targetAdvantageMean += currentTargetAdvantage;

                    if (maxTargetAdvantage > currentTargetAdvantage) continue;

                    maxTargetAdvantage = currentTargetAdvantage;
                }

                predictAdvantageMean /= _numberOfActions;
                targetAdvantageMean /= _numberOfActions;

                var experience = _experiences[_batchIndexes[i]];
                var experienceAction = experience.Action;

                _actionSample[i] =
                    _valuePredict[i, 0] + (_advantagePredict[i, experienceAction] - predictAdvantageMean);

                var qValueTarget = _valueTarget[i, 0] + (maxTargetAdvantage - targetAdvantageMean);
                _yTarget[i, experienceAction] =
                    experience.Done ? experience.Reward : experience.Reward + _gamma * qValueTarget;

                //for (int j = 0; j < _numberOfActions; j++) _dAdvantage[i, j] = 0.0f;
                //TODO: dividing by the number of actions might no longer make sense because we only updating one value 
                var dValue = -2.0f * (_yTarget[i, experienceAction] - _actionSample[i]) / _numberOfActions / _batchSize;
                _dAdvantage[i, experienceAction] = dValue * _duellingDerivative;

                // TODO: Derivative according to: https://datascience.stackexchange.com/questions/54023/dueling-network-gradient-with-respect-to-advantage-stream
                // float loss = -2.0f * (_yTarget[i, j] - _actionSample[i, j]) / _numberOfActions / _batchSize;
                // dValue += loss;
                // for (int k = 0; k < _numberOfActions; k++)
                // {
                //     _dAdvantage[i, k] += j == k ? loss * _duellingDerivative : loss * -0.1f;
                // }

                _dValue[i, 0] = dValue;
            }

            var dValueFinal = _duellingNetwork.ValueModel.Update(_dValue);
            var dAdvantageFinal = _duellingNetwork.AdvantageModel.Update(_dAdvantage);
            var dInputRowSize = _dInput.GetLength(1);
            for (int i = 0; i < _batchSize; i++)
            {
                for (int j = 0; j < dInputRowSize; j++)
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
                float error = _yTarget[i, _experiences[_batchIndexes[i]].Action] - _actionSample[i];
                loss += error * error / _numberOfActions;
            }

            loss /= _batchSize;
            return loss;
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
//     }
// }