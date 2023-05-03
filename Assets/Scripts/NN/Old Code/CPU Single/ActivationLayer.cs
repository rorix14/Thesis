using System.Linq;
using UnityEngine;

namespace NN.CPU_Single
{
    //TODO: this base class might not be needed, activation function can probably inherit directly form the BaseLayer class
    public abstract class ActivationFunction : BaseLayer
    {
        public abstract override void Forward(float[,] inputs);
        public abstract override void Backward(float[,] dValues);
    }

    public class ActivationReLu : ActivationFunction
    {
        public override void Forward(float[,] inputs)
        {
            Inputs = inputs;
            // TODO: there is no need to allocate memory for the output every time we do a forward pass, sizes will remain the same
            var inputsColumnSize = Inputs.GetLength(0);
            var inputsRowSize = Inputs.GetLength(1);
            
            Output = new float[inputsColumnSize, inputsRowSize];
            for (int i = 0; i < inputsColumnSize; i++)
            {
                for (int j = 0; j < inputsRowSize; j++)
                {
                    Output[i, j] = Inputs[i, j] <= 0 ? 0 : Inputs[i, j];
                }
            }
        }

        public override void Backward(float[,] dValues)
        {
            // float result = dValues.Cast<float>().Sum();
            // Debug.Log("(cpu) d_values value sum: " + result);
            DInputs = NnMath.CopyMatrix(dValues);
            for (int i = 0; i < DInputs.GetLength(0); i++)
            {
                for (int j = 0; j < DInputs.GetLength(1); j++)
                {
                    if (Inputs[i, j] <= 0)
                    {
                        DInputs[i, j] = 0;
                    }
                }
            }
        }
    }
    
    public class ActivationTanh : ActivationFunction
    {
        public override void Forward(float[,] inputs)
        {
            Output = new float[inputs.GetLength(0), inputs.GetLength(1)];
            for (int i = 0; i < inputs.GetLength(0); i++)
            {
                for (int j = 0; j < inputs.GetLength(1); j++)
                {
                    var input = inputs[i, j];
                    float exPos = Mathf.Exp(input);
                    float expNeg = Mathf.Exp(-input);
                    Output[i, j] = (exPos - expNeg) / (exPos + expNeg);
                }
            }
        }

        public override void Backward(float[,] dValues)
        {
            DInputs = NnMath.CopyMatrix(dValues);
            for (int i = 0; i < dValues.GetLength(0); i++)
            {
                for (int j = 0; j < dValues.GetLength(1); j++)
                {
                    DInputs[i, j] *= 1 - Output[i, j] * Output[i, j];
                }
            }
        }
    }
    
    public class ActivationLinear : ActivationFunction
    {
        public override void Forward(float[,] inputs)
        {
            Inputs = inputs;
            Output = inputs;
        }

        public override void Backward(float[,] dValues)
        {
            DInputs = NnMath.CopyMatrix(dValues);
        }
    }
}