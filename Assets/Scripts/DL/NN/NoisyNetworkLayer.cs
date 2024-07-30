using System;
using NN.CPU_Single;
using UnityEngine;
using Random = UnityEngine.Random;

namespace DL.NN
{
    public class NoisyNetworkLayer : NetworkLayer
    {
        // Neural network variables
        private readonly float[,] _sigmaWeights;
        private readonly float[,] _sigmaBiases;
        private readonly float[] _epsilonInputOutput;

        private readonly float[] _noiseSamplesBuffer;

        // Cashed variables
        private readonly int _epsilonArrayLength;
        private readonly int _noiseSamplesSize;

        // Compute buffer variables
        private readonly ComputeBuffer _sigmaWeightsBuffer;
        private readonly ComputeBuffer _sigmaBiasesBuffer;

        private readonly ComputeBuffer _epsilonInputOutputBuffer;

        private readonly int _kernelHandleBiasesBackward;
        private int _threadGroupXBiasesBackward;

        // Adam optimizer
        private ComputeBuffer _sigmaWeightsMomentumBuffer;
        private ComputeBuffer _sigmaWeightsCacheBuffer;
        private ComputeBuffer _sigmaBiasesMomentumBuffer;
        private ComputeBuffer _sigmaBiasesCacheBuffer;

        public NoisyNetworkLayer(int nInputs, int nNeurons, ActivationFunction activationFunction, ComputeShader shader,
            float sigma = 0.01f, bool isFirstLayer = false) : base(nInputs, nNeurons, activationFunction, shader,
            isFirstLayer, 1 / Mathf.Sqrt(nInputs), 1f /*0.01f*/)
        {
            _sigmaWeights = new float[nInputs, nNeurons];
            _sigmaBiases = new float[1, nNeurons];
            _epsilonInputOutput = new float[nInputs + nNeurons];
            _epsilonArrayLength = _epsilonInputOutput.Length;

            //var sigmaInitialValues = 0;
            var sigmaInitialValues = sigma / Mathf.Sqrt(nInputs);
            for (int i = 0; i < _weights.GetLength(1); i++)
            {
                _sigmaBiases[0, i] = sigmaInitialValues;
                for (int j = 0; j < _weights.GetLength(0); j++)
                {
                    _sigmaWeights[j, i] = sigmaInitialValues;
                }
            }

            switch (activationFunction)
            {
                case ActivationFunction.ReLu:
                    _kernelHandleBiasesBackward = _shader.FindKernel("backwards_pass_ReLU_biases_Adam");
                    break;
                case ActivationFunction.Tanh:
                    _kernelHandleBiasesBackward = _shader.FindKernel("backwards_pass_Tanh_biases_Adam");
                    break;
                case ActivationFunction.Linear:
                    _kernelHandleBiasesBackward = _shader.FindKernel("backwards_pass_linear_biases_Adam");
                    break;
            }

            _sigmaWeightsBuffer = new ComputeBuffer(_sigmaWeights.Length, sizeof(float));
            _sigmaBiasesBuffer = new ComputeBuffer(_sigmaBiases.Length, sizeof(float));

            _sigmaWeightsBuffer.SetData(_sigmaWeights);
            _sigmaBiasesBuffer.SetData(_sigmaBiases);

            _epsilonInputOutputBuffer = new ComputeBuffer(_epsilonInputOutput.Length, sizeof(float));

            _noiseSamplesSize = 10000000;
            _noiseSamplesBuffer = new float[_noiseSamplesSize];
            for (int i = 0; i < _noiseSamplesSize; i++)
            {
                var randomValue = NnMath.RandomGaussian(-4.0f, 4.0f);
                _noiseSamplesBuffer[i] = NnMath.Sign(randomValue) * Mathf.Sqrt(Mathf.Abs(randomValue));
            }
        }

        public override void Forward(Array input)
        {
            for (int i = 0; i < _epsilonArrayLength; i++)
            {
                _epsilonInputOutput[i] = _noiseSamplesBuffer[Random.Range(0, _noiseSamplesSize)];
            }

            _epsilonInputOutputBuffer.SetData(_epsilonInputOutput);
            base.Forward(input);
        }

        public override void Backward(Array dValue, float currentLearningRate)
        {
            base.Backward(dValue, currentLearningRate);
            _shader.Dispatch(_kernelHandleBiasesBackward, _threadGroupXBiasesBackward, 1, 1);
        }

        public override void CopyLayer(Layer otherLayer)
        {
            base.CopyLayer(otherLayer);

            var noisyLayer = (NoisyNetworkLayer)otherLayer;
            if (noisyLayer == null) return;

            _sigmaWeightsBuffer.GetData(noisyLayer._sigmaWeights);
            noisyLayer._sigmaWeightsBuffer.SetData(noisyLayer._sigmaWeights);

            _sigmaBiasesBuffer.GetData(noisyLayer._sigmaBiases);
            noisyLayer._sigmaBiasesBuffer.SetData(noisyLayer._sigmaBiases);
        }

        public override void Dispose()
        {
            base.Dispose();

            _sigmaWeightsBuffer?.Dispose();
            _sigmaBiasesBuffer?.Dispose();
            _epsilonInputOutputBuffer?.Dispose();

            _sigmaWeightsMomentumBuffer?.Dispose();
            _sigmaWeightsCacheBuffer?.Dispose();
            _sigmaBiasesMomentumBuffer?.Dispose();
            _sigmaBiasesCacheBuffer?.Dispose();
        }

        protected override void InitializeForwardBuffers(Array input)
        {
            base.InitializeForwardBuffers(input);

            _shader.SetBuffer(_kernelHandleForward, "sigma_weights", _sigmaWeightsBuffer);
            _shader.SetBuffer(_kernelHandleForward, "sigma_biases", _sigmaBiasesBuffer);
            _shader.SetBuffer(_kernelHandleForward, "epsilon_inputs_outputs", _epsilonInputOutputBuffer);
        }

        protected override void InitializeBackwardsBuffers(Array dValue)
        {
            base.InitializeBackwardsBuffers(dValue);

            _shader.GetKernelThreadGroupSizes(_kernelHandleBiasesBackward, out var x, out _, out _);
            _threadGroupXBiasesBackward = Mathf.CeilToInt(_biases.GetLength(1) / (float)x);

            _shader.SetBuffer(_kernelHandleInputsBackward, "sigma_weights", _sigmaWeightsBuffer);
            _shader.SetBuffer(_kernelHandleInputsBackward, "epsilon_inputs_outputs", _epsilonInputOutputBuffer);

            // Adam optimizer values
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "sigma_weights", _sigmaWeightsBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "epsilon_inputs_outputs", _epsilonInputOutputBuffer);

            _sigmaWeightsMomentumBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _sigmaWeightsCacheBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "sigma_weights_momentum",
                _sigmaWeightsMomentumBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "sigma_weights_cache", _sigmaWeightsCacheBuffer);

            var zeros = new float[_weights.Length];
            _sigmaWeightsMomentumBuffer.SetData(zeros);
            _sigmaWeightsCacheBuffer.SetData(zeros);

            _shader.SetBuffer(_kernelHandleBiasesBackward, "d_values", _dValuesBuffer);
            _shader.SetBuffer(_kernelHandleBiasesBackward, "biases", _biasesBuffer);
            _shader.SetBuffer(_kernelHandleBiasesBackward, "epsilon_inputs_outputs", _epsilonInputOutputBuffer);

            if (_activationFunction != ActivationFunction.Linear)
            {
                _shader.SetBuffer(_kernelHandleBiasesBackward, "output", _outputBuffer);
            }

            _shader.SetBuffer(_kernelHandleBiasesBackward, "biases_momentum", _biasesMomentumBuffer);
            _shader.SetBuffer(_kernelHandleBiasesBackward, "biases_cache", _biasesCacheBuffer);

            _shader.SetBuffer(_kernelHandleBiasesBackward, "sigma_biases", _sigmaBiasesBuffer);
            _sigmaBiasesMomentumBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _sigmaBiasesCacheBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _shader.SetBuffer(_kernelHandleBiasesBackward, "sigma_biases_momentum", _sigmaBiasesMomentumBuffer);
            _shader.SetBuffer(_kernelHandleBiasesBackward, "sigma_biases_cache", _sigmaBiasesCacheBuffer);

            zeros = new float [_biases.Length];
            _sigmaBiasesMomentumBuffer.SetData(zeros);
            _sigmaBiasesCacheBuffer.SetData(zeros);
        }
    }
}