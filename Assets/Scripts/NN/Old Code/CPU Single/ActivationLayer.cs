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
            Output = new float[Inputs.GetLength(0), Inputs.GetLength(1)];
            for (int i = 0; i < Inputs.GetLength(0); i++)
            {
                for (int j = 0; j < Inputs.GetLength(1); j++)
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