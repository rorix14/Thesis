// TODO: see if the math NN library needs to be in this name space

using NN.CPU_Single;
using UnityEngine;

namespace NN
{
    public enum ActivationFunction
    {
        ReLu,
        Tanh,
        Softmax,
        Linear
    }

    public class NetworkLayer
    {
        // Neural network variables
        public float[,] Output;
        public float[,] DInputs;
        protected readonly float[,] _weights;
        protected readonly float[,] _biases;

        protected readonly ActivationFunction _activationFunction;

        private readonly bool _isFirstLayer;
        private bool _forwardInitialized;
        private bool _backwardInitialized;
        private readonly int _headNumber;

        // Compute buffer variables
        protected readonly ComputeShader _shader;

        private ComputeBuffer _inputBuffer;
        protected ComputeBuffer _outputBuffer;
        protected readonly ComputeBuffer _weightsBuffer;
        protected readonly ComputeBuffer _biasesBuffer;

        // forward only variables
        protected readonly int _kernelHandleForward;
        private int _threadGroupXOutputForward;
        private int _threadGroupYOutputForward;
        private readonly int _kernelHandleForward2;
        private int _threadGroupYOutputForward2;

        //backwards variables
        protected ComputeBuffer _dValuesBuffer;
        private ComputeBuffer _dInputsBuffer;

        protected readonly int _kernelHandleInputsBackward;
        protected readonly int _kernelHandleWeightsBiasesBackward;

        private int _threadGroupXInputsBackward;
        private int _threadGroupYInputsBackward;
        private int _threadGroupXWeightsBackward;
        private int _threadGroupYWeightsBackward;

        // Adam optimizer
        private ComputeBuffer _weightsMomentumBuffer;
        private ComputeBuffer _weightsCacheBuffer;
        protected ComputeBuffer _biasesMomentumBuffer;
        protected ComputeBuffer _biasesCacheBuffer;

        public NetworkLayer(int nInputs, int nNeurons, ActivationFunction activationFunction, ComputeShader shader,
            bool isFirstLayer = false, float paramsRange = 4.0f, float paramsCoefficient = 0.01f, int headNumber = 1)
        {
            // Seed used to better reproduce results, usual seeds are 42, 50, 34
            Random.InitState(42);
            // neural networks standard init
            _weights = new float[nInputs, nNeurons];
            _biases = new float[1, nNeurons];
            _isFirstLayer = isFirstLayer;

            for (int i = 0; i < _weights.GetLength(1); i++)
            {
                _biases[0, i] = paramsCoefficient * NnMath.RandomGaussian(-paramsRange, paramsRange);
                for (int j = 0; j < _weights.GetLength(0); j++)
                {
                    _weights[j, i] =
                        paramsCoefficient * NnMath.RandomGaussian(-paramsRange, paramsRange); //* Random.value;
                }
            }

            // compute shader variables
            _shader = shader;
            _activationFunction = activationFunction;
            _headNumber = headNumber;
            switch (activationFunction)
            {
                case ActivationFunction.ReLu:
                    _kernelHandleForward = _shader.FindKernel("forward_pass_ReLU");
                    _kernelHandleInputsBackward = _shader.FindKernel("backwards_pass_ReLU_inputs");
                    _kernelHandleWeightsBiasesBackward = _shader.FindKernel("backwards_pass_ReLU_weights_biases_Adam");
                    break;
                case ActivationFunction.Tanh:
                    _kernelHandleForward = _shader.FindKernel("forward_pass_Tanh");
                    _kernelHandleInputsBackward = _shader.FindKernel("backwards_pass_Tanh_inputs");
                    _kernelHandleWeightsBiasesBackward = _shader.FindKernel("backwards_pass_Tanh_weights_biases_Adam");
                    break;
                case ActivationFunction.Softmax:
                    _kernelHandleForward = _shader.FindKernel("forward_pass_linear");
                    _kernelHandleForward2 = _shader.FindKernel("forward_pass_softmax");
                    _kernelHandleInputsBackward = _shader.FindKernel("backwards_pass_softmax_inputs");
                    _kernelHandleWeightsBiasesBackward =
                        _shader.FindKernel("backwards_pass_softmax_weights_biases_Adam");
                    break;
                case ActivationFunction.Linear:
                    _kernelHandleForward = _shader.FindKernel("forward_pass_linear");
                    _kernelHandleInputsBackward = _shader.FindKernel("backwards_pass_linear_inputs");
                    _kernelHandleWeightsBiasesBackward =
                        _shader.FindKernel("backwards_pass_linear_weights_biases_Adam");
                    break;
            }

            _shader.SetInt("input_row_size", _weights.GetLength(0));
            _shader.SetInt("weights_row_size", _weights.GetLength(1));

            _weightsBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _biasesBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _weightsBuffer.SetData(_weights);
            _biasesBuffer.SetData(_biases);
        }

        public virtual void Forward(float[,] inputs)
        {
            if (!_forwardInitialized)
            {
                InitializeForwardBuffers(inputs);
            }

            _inputBuffer.SetData(inputs);
            _shader.Dispatch(_kernelHandleForward, _threadGroupXOutputForward, _threadGroupYOutputForward, 1);

            if (_activationFunction == ActivationFunction.Softmax)
            {
                _shader.Dispatch(_kernelHandleForward2, _threadGroupXOutputForward, _threadGroupYOutputForward2, 1);
            }

            _outputBuffer.GetData(Output);
        }

        public virtual void Backward(float[,] dValues, float currentLearningRate, float beta1Corrected,
            float beta2Corrected)
        {
            if (!_backwardInitialized)
            {
                InitializeBackwardsBuffers(dValues);
            }

            _shader.SetFloat("current_learning_rate", currentLearningRate);
            _shader.SetFloat("beta_1_corrected", beta1Corrected);
            _shader.SetFloat("beta_2_corrected", beta2Corrected);

            _dValuesBuffer.SetData(dValues);

            if (!_isFirstLayer)
            {
                _shader.Dispatch(_kernelHandleInputsBackward, _threadGroupXInputsBackward,
                    _threadGroupYInputsBackward, 1);
                _dInputsBuffer.GetData(DInputs);
            }

            _shader.Dispatch(_kernelHandleWeightsBiasesBackward, _threadGroupXWeightsBackward,
                _threadGroupYWeightsBackward, 1);
            //_shader.Dispatch(_kernelHandleBiasesBackward, _threadGroupXBiasesBackward, 1, 1);
        }

        public void SetOptimizerVariables(float beta1, float beta2, float epsilon)
        {
            //TODO: can set set different optimizers based on a condition
            _shader.SetFloat("beta_1", beta1);
            _shader.SetFloat("beta_2", beta2);
            _shader.SetFloat("epsilon", epsilon);
        }

        public virtual void CopyLayer(NetworkLayer otherLayer)
        {
            _weightsBuffer.GetData(otherLayer._weights);
            otherLayer._weightsBuffer.SetData(otherLayer._weights);

            _biasesBuffer.GetData(otherLayer._biases);
            otherLayer._biasesBuffer.SetData(otherLayer._biases);

            // _weightsBuffer.GetData(_weights);
            // _biasesBuffer.GetData(_biases);
        }

        public virtual void Dispose()
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

        protected virtual void InitializeForwardBuffers(float[,] inputs)
        {
            Output = new float[inputs.GetLength(0), _weights.GetLength(1)];

            _shader.SetInt("input_column_size", inputs.GetLength(0));

            _shader.GetKernelThreadGroupSizes(_kernelHandleForward, out var threadSizeX, out var threadSizeY, out _);
            _threadGroupXOutputForward = Mathf.CeilToInt(Output.GetLength(0) / (float)threadSizeX);
            _threadGroupYOutputForward = Mathf.CeilToInt(Output.GetLength(1) / (float)threadSizeX);

            _shader.SetBuffer(_kernelHandleForward, "weights", _weightsBuffer);
            _shader.SetBuffer(_kernelHandleForward, "biases", _biasesBuffer);

            _inputBuffer = new ComputeBuffer(inputs.Length, sizeof(float));
            _outputBuffer = new ComputeBuffer(Output.Length, sizeof(float));
            _shader.SetBuffer(_kernelHandleForward, "input", _inputBuffer);
            _shader.SetBuffer(_kernelHandleForward, "output", _outputBuffer);

            if (_activationFunction == ActivationFunction.Softmax)
            {
                _threadGroupYOutputForward2 = Mathf.CeilToInt(_headNumber / (float)threadSizeY);
                _shader.SetInt("head_number", _headNumber);
                _shader.SetInt("distribution_lenght", _weights.GetLength(1) / _headNumber);
                _shader.SetBuffer(_kernelHandleForward2, "output", _outputBuffer);
            }

            _forwardInitialized = true;
        }

        protected virtual void InitializeBackwardsBuffers(float[,] dValues)
        {
            DInputs = new float[dValues.GetLength(0), _weights.GetLength(0)];

            _shader.GetKernelThreadGroupSizes(_kernelHandleInputsBackward, out var x, out var y, out _);
            _threadGroupXInputsBackward = Mathf.CeilToInt(DInputs.GetLength(0) / (float)x);
            _threadGroupYInputsBackward = Mathf.CeilToInt(DInputs.GetLength(1) / (float)y);

            _shader.GetKernelThreadGroupSizes(_kernelHandleWeightsBiasesBackward, out var threadSizeX,
                out var threadSizeY, out _);
            _threadGroupXWeightsBackward = Mathf.CeilToInt(_weights.GetLength(0) / (float)threadSizeX);
            _threadGroupYWeightsBackward = Mathf.CeilToInt(_weights.GetLength(1) / (float)threadSizeY);

            _shader.SetBuffer(_kernelHandleInputsBackward, "weights", _weightsBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "input", _inputBuffer);

            if (_activationFunction != ActivationFunction.Linear)
            {
                _shader.SetBuffer(_kernelHandleInputsBackward, "output", _outputBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "output", _outputBuffer);
            }

            _dValuesBuffer = new ComputeBuffer(dValues.Length, sizeof(float));
            _dInputsBuffer = new ComputeBuffer(DInputs.Length, sizeof(float));

            _shader.SetBuffer(_kernelHandleInputsBackward, "d_values", _dValuesBuffer);
            _shader.SetBuffer(_kernelHandleInputsBackward, "d_inputs", _dInputsBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "d_values", _dValuesBuffer);

            // Adam optimizer values
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "weights", _weightsBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "biases", _biasesBuffer);

            _weightsMomentumBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _weightsCacheBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "weights_momentum", _weightsMomentumBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "weights_cache", _weightsCacheBuffer);

            // init buffers with a matrix of zeros
            var zeros = new float[_weights.Length];
            _weightsMomentumBuffer.SetData(zeros);
            _weightsCacheBuffer.SetData(zeros);

            _biasesMomentumBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _biasesCacheBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "biases_momentum", _biasesMomentumBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "biases_cache", _biasesCacheBuffer);

            zeros = new float [_biases.Length];
            _biasesMomentumBuffer.SetData(zeros);
            _biasesCacheBuffer.SetData(zeros);

            _backwardInitialized = true;
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