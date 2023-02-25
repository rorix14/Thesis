using System.Linq;
using UnityEngine;

namespace NN.CPU_Single
{
    public abstract class LossFunction
    {
        public float[,] DInputs;
        
        public float RegularizationLoss(DenseLayer layer)
        {
            float regularizationLoss = 0;

            if (layer.WeightRegularizerL2 > 0)
            {
                float sum = 0;
                foreach (var weight in layer.Weights)
                {
                    sum += weight * weight;
                }

                regularizationLoss += layer.WeightRegularizerL2 * sum;
            }
            
            if (layer.BiasRegularizerL2 > 0)
            {
                float sum = 0;
                foreach (var bias in layer.Biases)
                {
                    sum += bias * bias;
                }

                regularizationLoss += layer.BiasRegularizerL2 * sum;
            }
            
            return regularizationLoss;
        }

        public float Calculate(float[,] output, float[,] y)
        {
            var sampleLosses = Forward(output, y);
            return NnMath.ArrayMean(sampleLosses);
        }

        public abstract float[] Forward(float[,] yPred, float[,] yTrue);

        public abstract void Backward(float[,] dValues, float[,] yTrue);
    }

    public class LossFunctionMeanSquaredError : LossFunction
    {
        public override float[] Forward(float[,] yPred, float[,] yTrue)
        {
            var sampleLosses = new float[yPred.GetLength(0)];
            for (int i = 0; i < yPred.GetLength(0); i++)
            {
                float sum = 0;
                float size = 0;
                for (int j = 0; j < yPred.GetLength(1); j++)
                {
                    sum += Mathf.Pow((yTrue[i, j] - yPred[i, j]), 2);
                    size = j;
                }

                sampleLosses[i] = sum / (size + 1);
            }

            return sampleLosses;
        }

        public override void Backward(float[,] dValues, float[,] yTrue)
        {
            var numOfSamples = dValues.GetLength(0);
            var numOfOutputs = dValues.GetLength(1);
            DInputs = new float[numOfSamples, numOfOutputs];

            for (int i = 0; i < numOfSamples; i++)
            {
                for (int j = 0; j < numOfOutputs; j++)
                {
                    DInputs[i, j] = -2 * ((yTrue[i, j] - dValues[i, j]) / numOfOutputs) / numOfSamples;
                }
            }
            
            // float result = DInputs.Cast<float>().Sum();
            // Debug.Log("(cpu) d_loss value sum: " + result);
        }
    }
}