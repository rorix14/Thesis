using System;
using UnityEngine;
using Random = UnityEngine.Random;

namespace DL.CNN
{
    public class ConvolutionalLayer : Layer
    {
        private readonly bool _isMaxPool;
        private readonly int _inputWidthHeight;

        // forward only variables
        private ComputeBuffer _outputNormalBuffer;

        private int _threadGroupXYForward;
        private int _threadGroupZForward;

        //backwards variables
        private ComputeBuffer _dOutputNormalBuffer;

        private readonly int _kernelHandleMaxPoolBackward;

        private int _threadGroupXYInputsBackward;
        private int _threadGroupZInputsBackward;
        private int _threadGroupXYWeightsBackward;
        private int _threadGroupZWeightsBackward;
        private int _threadGroupXYMaxPoolBackward;
        private int _threadGroupZMaxPoolBackward;

        // Assume valid type padding for all CNN operations

        public ConvolutionalLayer(int inputWidthHeight, int inputDepth, int filterSize, int featureMaps, int stride,
            bool maxPool, ComputeShader shader, bool isFirstLayer = false, float paramsRange = 4.0f,
            float paramsCoefficient = 0.005f) : base(shader, isFirstLayer)
        {
            var weightsTemp = new float[featureMaps, inputDepth, filterSize, filterSize];

            var outputSize = (inputWidthHeight - filterSize) / stride + 1;
            var biasesTemp = new float[featureMaps, outputSize, outputSize];

            //One common way to initialize filter weights is using HE initialization: Random.Range(-Mathf.Sqrt(2f / nInputs), Mathf.Sqrt(2f / nInputs));
            for (int i = 0; i < featureMaps; i++)
            {
                for (int by = 0; by < outputSize; by++)
                {
                    for (int bx = 0; bx < outputSize; bx++)
                    {
                        biasesTemp[i, by, bx] = paramsCoefficient * Random.Range(-paramsRange, paramsRange);
                    }
                }

                for (int j = 0; j < inputDepth; j++)
                {
                    for (int x = 0; x < filterSize; x++)
                    {
                        for (int y = 0; y < filterSize; y++)
                        {
                            weightsTemp[i, j, x, y] = paramsCoefficient * Random.Range(-paramsRange, paramsRange);
                        }
                    }
                }
            }

            _weights = weightsTemp;
            _biases = biasesTemp;

            _isMaxPool = maxPool;
            _inputWidthHeight = inputWidthHeight;

            if (_isMaxPool)
            {
                _kernelHandleForward = _shader.FindKernel("forward_pass_max_pool");
                _kernelHandleMaxPoolBackward = _shader.FindKernel("backward_pass_max_pool_output");
                _kernelHandleInputsBackward = _shader.FindKernel("backward_pass_max_pool_input");
                _kernelHandleWeightsBiasesBackward = _shader.FindKernel("backward_pass_max_pool_filter");
            }
            else
            {
                _kernelHandleForward = _shader.FindKernel("forward_pass");
                _kernelHandleInputsBackward = _shader.FindKernel("backward_pass_input");
                _kernelHandleWeightsBiasesBackward = _shader.FindKernel("backward_pass_filter");
            }

            _shader.SetInt("input_height_width", inputWidthHeight);
            _shader.SetInt("input_size", inputWidthHeight * inputWidthHeight);
            _shader.SetInt("input_element_size", inputDepth * inputWidthHeight * inputWidthHeight);

            _shader.SetInt("filter_stride", stride);
            _shader.SetInt("filter_number", _weights.GetLength(0));
            _shader.SetInt("filter_depth", _weights.GetLength(1));
            _shader.SetInt("filter_height_width", _weights.GetLength(2));
            _shader.SetInt("filter_size", _weights.GetLength(2) * _weights.GetLength(3));
            _shader.SetInt("filter_element_size",
                _weights.GetLength(1) * _weights.GetLength(2) * _weights.GetLength(3));

            _shader.SetInt("output_height_width", _biases.GetLength(1));
            _shader.SetInt("output_size", _biases.GetLength(1) * _biases.GetLength(2));
            _shader.SetInt("output_element_size", _biases.GetLength(0) * _biases.GetLength(1) * _biases.GetLength(2));

            if (_isMaxPool)
            {
                _shader.SetInt("output_max_pool_stride", 2);
                var maxPoolWidthHeight = outputSize / 2;
                _shader.SetInt("output_max_pool_height_width", maxPoolWidthHeight);
                _shader.SetInt("output_max_pool_size", maxPoolWidthHeight * maxPoolWidthHeight);
                shader.SetInt("output_max_pool_element_size",
                    _weights.GetLength(0) * maxPoolWidthHeight * maxPoolWidthHeight);
            }

            _weightsBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _biasesBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _weightsBuffer.SetData(_weights);
            _biasesBuffer.SetData(_biases);
        }
        
        public override void Forward(Array input)
        {
            if (!_forwardInitialized)
            {
                InitializeForwardBuffers(input);
            }

            _inputBuffer.SetData(input);
            _shader.Dispatch(_kernelHandleForward, _threadGroupXYForward, _threadGroupXYForward, _threadGroupZForward);
            _outputBuffer.GetData(Output);
        }

        public override void Backward(Array dValue, float currentLearningRate)
        {
            if (!_backwardInitialized)
            {
                InitializeBackwardsBuffers(dValue);
            }

            _shader.SetFloat(_currentLearningRateID, currentLearningRate);

            _dValuesBuffer.SetData(dValue);

            if (_isMaxPool)
            {
                _shader.Dispatch(_kernelHandleMaxPoolBackward, _threadGroupXYMaxPoolBackward,
                    _threadGroupXYMaxPoolBackward, _threadGroupZMaxPoolBackward);
            }

            if (!_isFirstLayer)
            {
                _shader.Dispatch(_kernelHandleInputsBackward, _threadGroupXYInputsBackward,
                    _threadGroupXYInputsBackward, _threadGroupZInputsBackward);
                _dInputsBuffer.GetData(DInput);
            }

            _shader.Dispatch(_kernelHandleWeightsBiasesBackward, _threadGroupXYWeightsBackward,
                _threadGroupXYWeightsBackward, _threadGroupZWeightsBackward);
        }

        public override void Dispose()
        {
            base.Dispose();

            _outputNormalBuffer?.Dispose();
            _dOutputNormalBuffer?.Dispose();
        }

        protected override void InitializeForwardBuffers(Array input)
        {
            var outputWidthHeight = _biases.GetLength(1);
            outputWidthHeight = _isMaxPool ? outputWidthHeight / 2 : outputWidthHeight;

            Output ??= new float[input.GetLength(0), _weights.GetLength(0), outputWidthHeight, outputWidthHeight];

            _shader.SetInt("output_number", Output.GetLength(0));

            _shader.GetKernelThreadGroupSizes(_kernelHandleForward, out var x, out _, out var z);
            _threadGroupXYForward = Mathf.CeilToInt(Output.GetLength(2) / (float)x);
            _threadGroupZForward = Mathf.CeilToInt(Output.GetLength(0) / (float)z);

            _shader.SetBuffer(_kernelHandleForward, "filter", _weightsBuffer);
            _shader.SetBuffer(_kernelHandleForward, "bias", _biasesBuffer);

            _inputBuffer = new ComputeBuffer(input.Length, sizeof(float));
            _outputBuffer = new ComputeBuffer(Output.Length, sizeof(float));
            _shader.SetBuffer(_kernelHandleForward, "input", _inputBuffer);
            _shader.SetBuffer(_kernelHandleForward, _isMaxPool ? "output_max_pool" : "output", _outputBuffer);

            if (_isMaxPool)
            {
                _outputNormalBuffer =
                    new ComputeBuffer(
                        input.GetLength(0) * _weights.GetLength(0) * _biases.GetLength(1) * _biases.GetLength(2),
                        sizeof(float));
                _shader.SetBuffer(_kernelHandleForward, "output", _outputNormalBuffer);
            }

            _forwardInitialized = true;
        }

        protected override void InitializeBackwardsBuffers(Array dValue)
        {
            DInput = new float[dValue.GetLength(0), _weights.GetLength(1), _inputWidthHeight, _inputWidthHeight];

            _shader.GetKernelThreadGroupSizes(_kernelHandleInputsBackward, out var inputX, out _, out var inputZ);
            _threadGroupXYInputsBackward = Mathf.CeilToInt(_inputWidthHeight / (float)inputX);
            _threadGroupZInputsBackward = Mathf.CeilToInt(DInput.GetLength(0) / (float)inputZ);

            _shader.GetKernelThreadGroupSizes(_kernelHandleWeightsBiasesBackward, out var weightX, out _,
                out var weightZ);
            _threadGroupXYWeightsBackward = Mathf.CeilToInt(_weights.GetLength(2) / (float)weightX);
            _threadGroupZWeightsBackward = Mathf.CeilToInt(_weights.GetLength(0) / (float)weightZ);

            _dValuesBuffer = new ComputeBuffer(dValue.Length, sizeof(float));
            _dInputsBuffer = new ComputeBuffer(DInput.Length, sizeof(float));

            _shader.SetBuffer(_kernelHandleInputsBackward, "filter", _weightsBuffer);
            _shader.SetBuffer(_kernelHandleInputsBackward, "d_inputs", _dInputsBuffer);

            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "input", _inputBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "filter", _weightsBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "bias", _biasesBuffer);
            
            if (_isMaxPool)
            {
                _shader.GetKernelThreadGroupSizes(_kernelHandleMaxPoolBackward, out var maxPoolX, out _,
                    out var maxPoolZ);
                _threadGroupXYMaxPoolBackward = Mathf.CeilToInt(Output.GetLength(2) / (float)maxPoolX);
                _threadGroupZMaxPoolBackward = Mathf.CeilToInt(Output.GetLength(0) / (float)maxPoolZ);

                _shader.SetBuffer(_kernelHandleMaxPoolBackward, "d_values", _dValuesBuffer);
                _shader.SetBuffer(_kernelHandleMaxPoolBackward, "output", _outputNormalBuffer);

                _dOutputNormalBuffer =
                    new ComputeBuffer(
                        dValue.GetLength(0) * _weights.GetLength(0) * _biases.GetLength(1) * _biases.GetLength(2),
                        sizeof(float));
                _shader.SetBuffer(_kernelHandleMaxPoolBackward, "d_output", _dOutputNormalBuffer);
                _shader.SetBuffer(_kernelHandleInputsBackward, "d_output", _dOutputNormalBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "d_output", _dOutputNormalBuffer);
            }
            else
            {
                _shader.SetBuffer(_kernelHandleInputsBackward, "d_values", _dValuesBuffer);
                _shader.SetBuffer(_kernelHandleInputsBackward, "output", _outputBuffer);

                _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "d_values", _dValuesBuffer);
                _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "output", _outputBuffer);
            }

            // Adam optimizer values
            _weightsMomentumBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _weightsCacheBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "filter_momentum", _weightsMomentumBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "filter_cache", _weightsCacheBuffer);

            var zeros = new float[_weights.Length];
            _weightsMomentumBuffer.SetData(zeros);
            _weightsCacheBuffer.SetData(zeros);

            _biasesMomentumBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _biasesCacheBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "bias_momentum", _biasesMomentumBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "bias_cache", _biasesCacheBuffer);

            zeros = new float [_biases.Length];
            _biasesMomentumBuffer.SetData(zeros);
            _biasesCacheBuffer.SetData(zeros);

            _backwardInitialized = true;
        }
    }
}