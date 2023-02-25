// TODO: see if the math NN library needs to be in this name space

using System.Linq;
using NN.CPU_Single;
using UnityEngine;

namespace NN
{
    public class NetworkLayer
    {
        // Neural network variables
        private float[,] _inputs;
        private readonly float[,] _weights;
        private readonly float[,] _biases;
        public float[,] Output;

        // Compute buffer variables
        private readonly ComputeShader _shader;

        private ComputeBuffer _inputBuffer;
        private ComputeBuffer _outputBuffer;
        private readonly ComputeBuffer _weightsBuffer;
        private readonly ComputeBuffer _biasesBuffer;

        private readonly int _kernelHandleForward;
        private readonly uint _threadSizeXForward;
        private readonly uint _threadSizeYForward;
        private int _threadGroupXForward;
        private int _threadGroupYForward;

        public NetworkLayer(int nInputs, int nNeurons, ComputeShader shader)
        {
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

            _shader = shader;
            _kernelHandleForward = _shader.FindKernel("forward_pass_ReLU");
            _shader.GetKernelThreadGroupSizes(_kernelHandleForward, out _threadSizeXForward, out _threadSizeYForward,
                out _);

            _shader.SetInt("input_row_size", _weights.GetLength(0));
            _shader.SetInt("weights_row_size", _weights.GetLength(1));

            _weightsBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _biasesBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _shader.SetBuffer(_kernelHandleForward, "weights", _weightsBuffer);
            _shader.SetBuffer(_kernelHandleForward, "biases", _biasesBuffer);
            _weightsBuffer.SetData(_weights);
            _biasesBuffer.SetData(_biases);
        }

        public void Forward(float[,] inputs)
        {
            if (_inputBuffer is null)
            {
                _shader.SetInt("input_column_size", inputs.GetLength(0));

                Output = new float[inputs.GetLength(0), _weights.GetLength(1)];

                _threadGroupXForward = Mathf.CeilToInt(Output.GetLength(0) / (float)_threadSizeXForward);
                _threadGroupYForward = Mathf.CeilToInt(Output.GetLength(1) / (float)_threadSizeYForward);

                _inputBuffer = new ComputeBuffer(inputs.Length, sizeof(float));
                _outputBuffer = new ComputeBuffer(Output.Length, sizeof(float));
                _shader.SetBuffer(_kernelHandleForward, "input", _inputBuffer);
                _shader.SetBuffer(_kernelHandleForward, "output", _outputBuffer);
            }

            _inputBuffer.SetData(inputs);
            _shader.Dispatch(_kernelHandleForward, _threadGroupXForward, _threadGroupYForward, 1);
            _outputBuffer.GetData(Output);
        }

        private ComputeBuffer _yTrueBuffer;

        public float ForwardLoss(float[,] yTrue)
        {
            int kernelHandleLoss = _shader.FindKernel("forward_pass_MSE_loss");
            _shader.GetKernelThreadGroupSizes(kernelHandleLoss, out var threadSizeX, out _, out _);
            var threadGroupX = Mathf.CeilToInt(yTrue.GetLength(0) / (float)threadSizeX);

            var sampleLossesBuffer = new ComputeBuffer(yTrue.GetLength(0), sizeof(float));
            _shader.SetBuffer(kernelHandleLoss, "sample_losses", sampleLossesBuffer);

            if (_yTrueBuffer is null)
            {
                _yTrueBuffer = new ComputeBuffer(yTrue.Length, sizeof(float));
                _shader.SetBuffer(kernelHandleLoss, "y_true", _yTrueBuffer);
                _shader.SetBuffer(kernelHandleLoss, "output", _outputBuffer);
            }

            _yTrueBuffer.SetData(yTrue);
            _shader.Dispatch(kernelHandleLoss, threadGroupX, 1, 1);
            var sampleLosses = new float[yTrue.GetLength(0)];
            sampleLossesBuffer.GetData(sampleLosses);

            float mean = 0.0f;
            foreach (var sampleLoss in sampleLosses)
                mean += sampleLoss;

            mean /= sampleLosses.Length;
            //Debug.Log("(GPU) loss: " + mean);

            sampleLossesBuffer.Dispose();
            return mean;
        }

        //public float[,] _dWeights;
        //public float[,] _dBiases;
        public float[,] DInputs;

        private ComputeBuffer _dValuesBuffer;
        private ComputeBuffer _dInputsBuffer;
        //private ComputeBuffer _dWeightsBuffer;
        //private ComputeBuffer _dBiasesBuffer;

        private int _kernelHandleWeightsBackward;
        private int _kernelHandleBiasesBackward;
        private int _kernelHandleInputsBackward;

        private uint _threadSizeXBackward;
        private uint _threadSizeYBackward;
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

        public float[,] DInputsLoss;

        public void BackwardLoss()
        {
            int kernelHandleLoss = _shader.FindKernel("backwards_pass_MSE_loss");
            _shader.GetKernelThreadGroupSizes(kernelHandleLoss, out var threadSizeX, out var threadSizeY, out _);
            var threadGroupX = Mathf.CeilToInt(Output.GetLength(0) / (float)threadSizeX);
            var threadGroupY = Mathf.CeilToInt(Output.GetLength(1) / (float)threadSizeY);

            DInputsLoss = new float[Output.GetLength(0), Output.GetLength(1)];
            var dInputsLossBuffer = new ComputeBuffer(Output.Length, sizeof(float));

            _shader.SetBuffer(kernelHandleLoss, "y_true", _yTrueBuffer);
            _shader.SetBuffer(kernelHandleLoss, "output", _outputBuffer);
            _shader.SetBuffer(kernelHandleLoss, "d_inputs_loss", dInputsLossBuffer);

            _shader.Dispatch(kernelHandleLoss, threadGroupX, threadGroupY, 1);
            dInputsLossBuffer.GetData(DInputsLoss);

            dInputsLossBuffer.Dispose();

            //float result = DInputsLoss.Cast<float>().Sum();
            //Debug.Log("(GPU) d_loss value sum: " + result);
        }

        private int _iteration = 0;
        private float _learningRate = 0.005f;
        private float _decay = 1e-3f;

        public void Backward(float[,] dValues)
        {
            if (DInputs is null)
            {
                //_dWeights = new float[_weights.GetLength(0), _weights.GetLength(1)];
                //_dBiases = new float[_biases.GetLength(0), _biases.GetLength(1)];
                DInputs = new float[dValues.GetLength(0), _weights.GetLength(0)];

                _kernelHandleWeightsBackward = _shader.FindKernel("backwards_pass_ReLU_weights_Adam");
                _kernelHandleBiasesBackward = _shader.FindKernel("backwards_pass_ReLU_biases_Adam");
                _kernelHandleInputsBackward = _shader.FindKernel("backwards_pass_ReLU_inputs");

                _dValuesBuffer = new ComputeBuffer(dValues.Length, sizeof(float));
                _dInputsBuffer = new ComputeBuffer(DInputs.Length, sizeof(float));
                // _dWeightsBuffer = new ComputeBuffer(_dWeights.Length, sizeof(float));
               // _dBiasesBuffer = new ComputeBuffer(_dBiases.Length, sizeof(float));

                _shader.SetBuffer(_kernelHandleInputsBackward, "d_values", _dValuesBuffer);
                _shader.SetBuffer(_kernelHandleInputsBackward, "d_inputs", _dInputsBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBackward, "d_values", _dValuesBuffer);
                // _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "d_weights", _dWeightsBuffer);
                _shader.SetBuffer(_kernelHandleBiasesBackward, "d_values", _dValuesBuffer);
                //_shader.SetBuffer(_kernelHandleBiasesBackward, "d_biases", _dBiasesBuffer);

                _shader.SetBuffer(_kernelHandleInputsBackward, "weights", _weightsBuffer);
                _shader.SetBuffer(_kernelHandleInputsBackward, "output", _outputBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBackward, "input", _inputBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBackward, "output", _outputBuffer);
                _shader.SetBuffer(_kernelHandleBiasesBackward, "output", _outputBuffer);

                _shader.GetKernelThreadGroupSizes(_kernelHandleInputsBackward, out var x, out var y, out _);
                _threadGroupXInputsBackward = Mathf.CeilToInt(DInputs.GetLength(0) / (float)x);
                _threadGroupYInputsBackward = Mathf.CeilToInt(DInputs.GetLength(1) / (float)y);

                _shader.GetKernelThreadGroupSizes(_kernelHandleWeightsBackward, out _threadSizeXBackward,
                    out _threadSizeYBackward, out _);
                _threadGroupXWeightsBackward = Mathf.CeilToInt(_weights.GetLength(0) / (float)_threadSizeXBackward);
                _threadGroupYWeightsBackward = Mathf.CeilToInt(_weights.GetLength(1) / (float)_threadSizeYBackward);

                _shader.GetKernelThreadGroupSizes(_kernelHandleBiasesBackward, out var biasX, out _, out _);
                _threadGroupXBiasesBackward = Mathf.CeilToInt(DInputs.GetLength(0) / (float)biasX);

                // Adam optimizer values
                _shader.SetFloat("beta_1", 0.9f);
                _shader.SetFloat("beta_2", 0.999f);
                _shader.SetFloat("epsilon", 1e-7f);

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

            _shader.SetInt("iteration", _iteration);
            _shader.SetFloat("learning_rate", _learningRate * (1.0f / (1.0f + _decay * _iteration)));

            _dValuesBuffer.SetData(dValues);
            _shader.Dispatch(_kernelHandleInputsBackward, _threadGroupXInputsBackward,
                _threadGroupYInputsBackward, 1);

            _shader.Dispatch(_kernelHandleWeightsBackward, _threadGroupXWeightsBackward,
                _threadGroupYWeightsBackward, 1);
            _shader.Dispatch(_kernelHandleBiasesBackward, _threadGroupXBiasesBackward, 1, 1);

            _dInputsBuffer.GetData(DInputs);

            ++_iteration;
        }

        public void Dispose()
        {
            _yTrueBuffer?.Dispose();
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
            //_dWeightsBuffer?.Dispose();
            //_dBiasesBuffer?.Dispose();
        }
    }
}