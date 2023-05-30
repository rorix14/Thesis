using NN;
using NN.CPU_Single;

namespace Algorithms.NE
{
    public class ESModel : NetworkModel
    {
        public ESModel(NetworkLayer[] layers, NetworkLoss lossFunction, float learningRate = 0.005f,
            float decay = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float epsilon = 1E-07f) : base(layers,
            lossFunction, learningRate, decay, beta1, beta2, epsilon)
        {
            
        }

        public override float[,] Update(float[,] yTarget)
        {
            //TODO: mean is already calculated in the model, so it can be just passed
            float rewardMean = NnMath.MatrixMean(yTarget);
            float rewardStd = NnMath.StandardDivination(yTarget, rewardMean);
            
            for (int i = 0; i < _layers.Length; i++)
            {
                _layers[i].Backward(yTarget, _currentLearningRate, rewardMean, rewardStd);
            }
            
            return null;
        }
    }
}