using NN;

namespace Algorithms.RL
{
    public class NoisyDQN : ModelDQN
    {
        public NoisyDQN(NetworkModel networkModel, NetworkModel targetModel, int numberOfActions, int stateSize,
            int maxExperienceSize = 10000, int minExperienceSize = 100, int batchSize = 32, float gamma = 0.99f) : base(
            networkModel, targetModel, numberOfActions, stateSize, maxExperienceSize, minExperienceSize, batchSize,
            gamma)
        {
            _predictSate = new float[batchSize, stateSize];
        }

        public override int EpsilonGreedySample(float[] state, float eps = 0.1f)
        {
            for (int i = 0; i < _stateLenght; i++)
            {
                _predictSate[0, i] = state[i];
            }

            MaxByRow(_networkModel.Predict(_predictSate), true);
            return _nextQ[0].index;
        }
    }
}