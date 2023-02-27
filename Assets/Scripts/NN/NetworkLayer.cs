// TODO: see if the math NN library needs to be in this name space
using NN.CPU_Single;
using UnityEngine;
using Random = UnityEngine.Random;

namespace NN
{
    public enum ActivationFunction
    {
        ReLu,
        Linear
    }

    public class NetworkLayer
    {
        // Neural network variables
        public float[,] Output;
        public float[,] DInputs;
        private readonly float[,] _weights;
        private readonly float[,] _biases;
        
        // Compute buffer variables
        private readonly ComputeShader _shader;
        
        private ComputeBuffer _inputBuffer;
        private ComputeBuffer _outputBuffer;
        private readonly ComputeBuffer _weightsBuffer;
        private readonly ComputeBuffer _biasesBuffer;
        
        // forward only variables
        private readonly int _kernelHandleForward;
        private int _threadGroupXOutputForward;
        private int _threadGroupYOutputForward;
        
        //backwards variables
        private ComputeBuffer _dValuesBuffer;
        private ComputeBuffer _dInputsBuffer;
        
        private readonly int _kernelHandleWeightsBackward;
        private readonly int _kernelHandleBiasesBackward;
        private readonly int _kernelHandleInputsBackward;
        
        private int _threadGroupXWeightsBackward;
        private int _threadGroupYWeightsBackward;
        private int _threadGroupXBiasesBackward;
        private int _threadGroupXInputsBackward;
        private int _threadGroupYInputsBackward;

        // Adam optimizer
        private ComputeBuffer _weightsMomentumBuffer;
        private ComputeBuffer _weightsCacheBuffer;
        private ComputeBuffer _biasesMomentumBuffer;
        private ComputeBuffer _biasesCacheBuffer;
        
        public NetworkLayer(int nInputs, int nNeurons, ActivationFunction activationFunction, ComputeShader shader)
        {
            // neural networks standard init
            Random.InitState(42);
            _weights = new float[nInputs, nNeurons];
            _biases = new float[1, nNeurons];

            for (int i = 0; i < _weights.GetLength(1); i++)
            {
                _biases[0, i] = 0.01f * NnMath.RandomGaussian(-4.0f, 4.0f);
                for (int j = 0; j < _weights.GetLength(0); j++)
                {
                    _weights[j, i] = 0.01f * NnMath.RandomGaussian(-4.0f, 4.0f); //* Random.value;
                }
            }

            // compute shader variables
            _shader = shader;

            // TODO: this should be a function
            switch (activationFunction)
            {
                case ActivationFunction.ReLu:
                    _kernelHandleForward = _shader.FindKernel("forward_pass_ReLU");
                    _kernelHandleInputsBackward = _shader.FindKernel("backwards_pass_ReLU_inputs");
                    _kernelHandleWeightsBackward = _shader.FindKernel("backwards_pass_ReLU_weights_Adam");
                    _kernelHandleBiasesBackward = _shader.FindKernel("backwards_pass_ReLU_biases_Adam");
                    break;
                case ActivationFunction.Linear:
                    _kernelHandleForward = _shader.FindKernel("forward_pass_linear");
                    _kernelHandleInputsBackward = _shader.FindKernel("backwards_pass_Linear_inputs");
                    _kernelHandleWeightsBackward = _shader.FindKernel("backwards_pass_linear_weights_Adam");
                    _kernelHandleBiasesBackward = _shader.FindKernel("backwards_pass_linear_biases_Adam");
                    break;
            }
            
            _shader.SetInt("input_row_size", _weights.GetLength(0));
            _shader.SetInt("weights_row_size", _weights.GetLength(1));

            _weightsBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _biasesBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            
            _weightsBuffer.SetData(_weights);
            _biasesBuffer.SetData(_biases);
        }

        public void Forward(float[,] inputs)
        {
            if (Output is null)
            {
                Output = new float[inputs.GetLength(0), _weights.GetLength(1)];

                _shader.SetInt("input_column_size", inputs.GetLength(0));

                _shader.GetKernelThreadGroupSizes(_kernelHandleForward, out var threadSizeX, out var threadSizeY,
                    out _);
                _threadGroupXOutputForward = Mathf.CeilToInt(Output.GetLength(0) / (float)threadSizeX);
                _threadGroupYOutputForward = Mathf.CeilToInt(Output.GetLength(1) / (float)threadSizeY);
                
                _shader.SetBuffer(_kernelHandleForward, "weights", _weightsBuffer);
                _shader.SetBuffer(_kernelHandleForward, "biases", _biasesBuffer);

                _inputBuffer = new ComputeBuffer(inputs.Length, sizeof(float));
                _outputBuffer = new ComputeBuffer(Output.Length, sizeof(float));
                _shader.SetBuffer(_kernelHandleForward, "input", _inputBuffer);
                _shader.SetBuffer(_kernelHandleForward, "output", _outputBuffer);
            }

            _inputBuffer.SetData(inputs);
            _shader.Dispatch(_kernelHandleForward, _threadGroupXOutputForward, _threadGroupYOutputForward, 1);
            _outputBuffer.GetData(Output);
        }
        
        public void Backward(float[,] dValues, float learningRate, int iteration)
        {
            if (DInputs is null)
            {
                DInputs = new float[dValues.GetLength(0), _weights.GetLength(0)];

                _shader.GetKernelThreadGroupSizes(_kernelHandleInputsBackward, out var x, out var y, out _);
                _threadGroupXInputsBackward = Mathf.CeilToInt(DInputs.GetLength(0) / (float)x);
                _threadGroupYInputsBackward = Mathf.CeilToInt(DInputs.GetLength(1) / (float)y);
                
                _shader.GetKernelThreadGroupSizes(_kernelHandleWeightsBackward, out var threadSizeX,
                    out var threadSizeY, out _);
                _threadGroupXWeightsBackward = Mathf.CeilToInt(_weights.GetLength(0) / (float)threadSizeX);
                _threadGroupYWeightsBackward = Mathf.CeilToInt(_weights.GetLength(1) / (float)threadSizeY);

                _shader.GetKernelThreadGroupSizes(_kernelHandleBiasesBackward, out var biasX, out _, out _);
                _threadGroupXBiasesBackward = Mathf.CeilToInt(DInputs.GetLength(0) / (float)biasX);
                
                _shader.SetBuffer(_kernelHandleInputsBackward, "weights", _weightsBuffer);
                _shader.SetBuffer(_kernelHandleInputsBackward, "output", _outputBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBackward, "input", _inputBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBackward, "output", _outputBuffer);
                _shader.SetBuffer(_kernelHandleBiasesBackward, "output", _outputBuffer);
                
                _dValuesBuffer = new ComputeBuffer(dValues.Length, sizeof(float));
                _dInputsBuffer = new ComputeBuffer(DInputs.Length, sizeof(float));

                _shader.SetBuffer(_kernelHandleInputsBackward, "d_values", _dValuesBuffer);
                _shader.SetBuffer(_kernelHandleInputsBackward, "d_inputs", _dInputsBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBackward, "d_values", _dValuesBuffer);
                _shader.SetBuffer(_kernelHandleBiasesBackward, "d_values", _dValuesBuffer);
                
                // Adam optimizer values
                _shader.SetBuffer(_kernelHandleWeightsBackward, "weights", _weightsBuffer);
                _shader.SetBuffer(_kernelHandleBiasesBackward, "biases", _biasesBuffer);

                _weightsMomentumBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
                _weightsCacheBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
                _shader.SetBuffer(_kernelHandleWeightsBackward, "weights_momentum", _weightsMomentumBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBackward, "weights_cache", _weightsCacheBuffer);
                
                // init buffers with a matrix of zeros
                var zeros = new float[_weights.Length];
                _weightsMomentumBuffer.SetData(zeros);
                _weightsCacheBuffer.SetData(zeros);

                _biasesMomentumBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
                _biasesCacheBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
                
                _shader.SetBuffer(_kernelHandleBiasesBackward, "biases_momentum", _biasesMomentumBuffer);
                _shader.SetBuffer(_kernelHandleBiasesBackward, "biases_cache", _biasesCacheBuffer);
                
                zeros = new float [_biases.Length];
                _biasesMomentumBuffer.SetData(zeros);
                _biasesCacheBuffer.SetData(zeros);
            }

            _shader.SetFloat("learning_rate", learningRate);
            _shader.SetInt("iteration", iteration);

            _dValuesBuffer.SetData(dValues);
            _shader.Dispatch(_kernelHandleInputsBackward, _threadGroupXInputsBackward,
                _threadGroupYInputsBackward, 1);

            _shader.Dispatch(_kernelHandleWeightsBackward, _threadGroupXWeightsBackward,
                _threadGroupYWeightsBackward, 1);
            _shader.Dispatch(_kernelHandleBiasesBackward, _threadGroupXBiasesBackward, 1, 1);

            _dInputsBuffer.GetData(DInputs);
        }

        public void SetOptimizerVariables(float beta1, float beta2, float epsilon)
        {
            _shader.SetFloat("beta_1", beta1);
            _shader.SetFloat("beta_2", beta2);
            _shader.SetFloat("epsilon", epsilon);
        }

        public void Dispose()
        {
            _weightsMomentumBuffer?.Dispose();
            _weightsCacheBuffer?.Dispose();
            _biasesMomentumBuffer?.Dispose();
            _biasesCacheBuffer?.Dispose();

            _inputBuffer?.Dispose();
            _outputBuffer?.Dispose();
            _weightsBuffer?.Dispose();
            _biasesBuffer?.Dispose();

            _dValuesBuffer?.Dispose();
            _dInputsBuffer?.Dispose();
        }

        // private ComputeBuffer _yTrueBuffer;
        // public float ForwardLoss(float[,] yTrue)
        // {
        //     int kernelHandleLoss = _shader.FindKernel("forward_pass_MSE_loss");
        //     _shader.GetKernelThreadGroupSizes(kernelHandleLoss, out var threadSizeX, out _, out _);
        //     var threadGroupX = Mathf.CeilToInt(yTrue.GetLength(0) / (float)threadSizeX);
        //
        //     var sampleLossesBuffer = new ComputeBuffer(yTrue.GetLength(0), sizeof(float));
        //     _shader.SetBuffer(kernelHandleLoss, "sample_losses", sampleLossesBuffer);
        //
        //     if (_yTrueBuffer is null)
        //     {
        //         _yTrueBuffer = new ComputeBuffer(yTrue.Length, sizeof(float));
        //         _shader.SetBuffer(kernelHandleLoss, "y_true", _yTrueBuffer);
        //         _shader.SetBuffer(kernelHandleLoss, "output", _outputBuffer);
        //     }
        //
        //     _yTrueBuffer.SetData(yTrue);
        //     _shader.Dispatch(kernelHandleLoss, threadGroupX, 1, 1);
        //     var sampleLosses = new float[yTrue.GetLength(0)];
        //     sampleLossesBuffer.GetData(sampleLosses);
        //
        //     float mean = 0.0f;
        //     foreach (var sampleLoss in sampleLosses)
        //         mean += sampleLoss;
        //
        //     mean /= sampleLosses.Length;
        //
        //     sampleLossesBuffer.Dispose();
        //     return mean;
        // }

        // public float[,] DInputsLoss;
        // public void BackwardLoss()
        // {
        //     int kernelHandleLoss = _shader.FindKernel("backwards_pass_MSE_loss");
        //     _shader.GetKernelThreadGroupSizes(kernelHandleLoss, out var threadSizeX, out var threadSizeY, out _);
        //     var threadGroupX = Mathf.CeilToInt(Output.GetLength(0) / (float)threadSizeX);
        //     var threadGroupY = Mathf.CeilToInt(Output.GetLength(1) / (float)threadSizeY);
        //
        //     DInputsLoss = new float[Output.GetLength(0), Output.GetLength(1)];
        //     var dInputsLossBuffer = new ComputeBuffer(Output.Length, sizeof(float));
        //
        //     _shader.SetBuffer(kernelHandleLoss, "y_true", _yTrueBuffer);
        //     _shader.SetBuffer(kernelHandleLoss, "output", _outputBuffer);
        //     _shader.SetBuffer(kernelHandleLoss, "d_inputs_loss", dInputsLossBuffer);
        //
        //     _shader.Dispatch(kernelHandleLoss, threadGroupX, threadGroupY, 1);
        //     dInputsLossBuffer.GetData(DInputsLoss);
        //
        //     dInputsLossBuffer.Dispose();
        // }
    }
}