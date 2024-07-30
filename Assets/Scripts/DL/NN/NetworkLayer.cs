// TODO: see if the math NN library needs to be in this name space
using System;
using UnityEngine;
using Random = UnityEngine.Random;

namespace DL.NN
{
    public class NetworkLayer : Layer
    {
        protected readonly ActivationFunction _activationFunction;

        private readonly int _headNumber;

        // forward only variables
        private int _threadGroupXOutputForward;
        private int _threadGroupYOutputForward;

        private readonly int _kernelHandleForward2;
        private int _threadGroupYOutputForward2;

        //backwards variables
        protected int _threadGroupXInputsBackward;
        protected int _threadGroupYInputsBackward;
        protected int _threadGroupXWeightsBackward;
        protected int _threadGroupYWeightsBackward;


        //TODO: erase, oly uses for testing peruses
        // public ComputeBuffer weightsTestBuffer;

        public NetworkLayer(int nInputs, int nNeurons, ActivationFunction activationFunction, ComputeShader shader,
            bool isFirstLayer = false, float paramsRange = 4.0f, float paramsCoefficient = 0.005f,
            int headNumber = 1) : base(shader, isFirstLayer)
        {
            // Seed used to better reproduce results, usual seeds are 42, 50, 34
            //Random.InitState(42);
            // neural networks standard init
            var weightsTemp = new float[nInputs, nNeurons];
            var biasesTemp = new float[1, nNeurons];

            //paramsRange = 1f / Mathf.Sqrt(nInputs);
            //paramsRange = Mathf.Sqrt(6) / Mathf.Sqrt(nInputs + nNeurons);
            //paramsCoefficient = Mathf.Sqrt(2f / nInputs);

            for (int i = 0; i < weightsTemp.GetLength(1); i++)
            {
                // _biases[0, i] = 0.1f;
                //_biases[0, i] = Random.Range(-paramsRange, paramsRange);
                biasesTemp[0, i] =
                    paramsCoefficient *
                    Random.Range(-paramsRange, paramsRange) /*NnMath.RandomGaussian(-paramsRange, paramsRange)*/;
                for (int j = 0; j < weightsTemp.GetLength(0); j++)
                {
                    // _weights[j, i] = 0.01f;
                    //_weights[j, i] = Random.Range(-paramsRange, paramsRange);
                    weightsTemp[j, i] =
                        paramsCoefficient *
                        Random.Range(-paramsRange, paramsRange) /*NnMath.RandomGaussian(-paramsRange, paramsRange)*/;
                }
            }
            _weights = weightsTemp;
            _biases = biasesTemp;

            // compute shader variables
            _headNumber = headNumber;
            _activationFunction = activationFunction;
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

            // weightsTestBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            // weightsTestBuffer.SetData(new float[_weights.Length]);
            // _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "weights_test", weightsTestBuffer);
        }

        public override void Forward(Array input)
        {
            if (!_forwardInitialized)
            {
                InitializeForwardBuffers(input);
            }

            _inputBuffer.SetData(input);
            _shader.Dispatch(_kernelHandleForward, _threadGroupXOutputForward, _threadGroupYOutputForward, 1);

            if (_activationFunction == ActivationFunction.Softmax)
            {
                _shader.Dispatch(_kernelHandleForward2, _threadGroupXOutputForward, _threadGroupYOutputForward2, 1);
            }

            _outputBuffer.GetData(Output);
        }

        public override void Backward(Array dValue, float currentLearningRate)
        {
            if (!_backwardInitialized)
            {
                InitializeBackwardsBuffers(dValue);
            }

            _shader.SetFloat(_currentLearningRateID, currentLearningRate);

            // for (int i = 0; i < dValues.GetLength(0); i++)
            // {
            //     for (int j = 0; j < dValues.GetLength(1); j++)
            //     {
            //         dValues[i, j] = Random.Range(5, 1000);
            //     }
            // }
            _dValuesBuffer.SetData(dValue);

            if (!_isFirstLayer)
            {
                _shader.Dispatch(_kernelHandleInputsBackward, _threadGroupXInputsBackward,
                    _threadGroupYInputsBackward, 1);
                _dInputsBuffer.GetData(DInput);
            }

            _shader.Dispatch(_kernelHandleWeightsBiasesBackward, _threadGroupXWeightsBackward,
                _threadGroupYWeightsBackward, 1);
            //_shader.Dispatch(_kernelHandleBiasesBackward, _threadGroupXBiasesBackward, 1, 1);

            // _weightsBuffer.GetData(_weights);
            // _biasesBuffer.GetData(_biases);
            // var weightMomentum = new float[_weights.GetLength(0), _weights.GetLength(1)];
            // var weightCache = new float[_weights.GetLength(0), _weights.GetLength(1)];
            // _weightsMomentumBuffer.GetData(weightMomentum);
            // _weightsCacheBuffer.GetData(weightCache);
            //
            // var biasMomentum = new float[_biases.GetLength(0), _biases.GetLength(1)];
            // var biasCache = new float[_biases.GetLength(0), _biases.GetLength(1)];
            // _biasesMomentumBuffer.GetData(biasMomentum);
            // _biasesCacheBuffer.GetData(biasCache);
        }

        protected override void InitializeForwardBuffers(Array inputs)
        {
            Output ??= new float[inputs.GetLength(0), _weights.GetLength(1)];

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
                _shader.SetInt("distribution_length", _weights.GetLength(1) / _headNumber);
                _shader.SetBuffer(_kernelHandleForward2, "output", _outputBuffer);
            }

            _forwardInitialized = true;
        }

        protected override void InitializeBackwardsBuffers(Array dValue)
        {
            DInput = new float[dValue.GetLength(0), _weights.GetLength(0)];

            _shader.GetKernelThreadGroupSizes(_kernelHandleInputsBackward, out var x, out var y, out _);
            _threadGroupXInputsBackward = Mathf.CeilToInt(DInput.GetLength(0) / (float)x);
            _threadGroupYInputsBackward = Mathf.CeilToInt(DInput.GetLength(1) / (float)y);

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

            _dValuesBuffer = new ComputeBuffer(dValue.Length, sizeof(float));
            _dInputsBuffer = new ComputeBuffer(DInput.Length, sizeof(float));

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