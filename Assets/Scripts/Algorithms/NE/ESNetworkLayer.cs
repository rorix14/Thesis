using NN;
using NN.CPU_Single;
using UnityEngine;
using ActivationFunction = NN.ActivationFunction;

namespace Algorithms.NE
{
    public class ESNetworkLayer : NetworkLayer
    {
        private float _noiseStD;
        private float _learningRate;

        private readonly float[,] _weightNoise;
        private readonly float[] _biasNoise;

        private readonly ComputeBuffer _weightNoiseBuffer;
        private readonly ComputeBuffer _biasNoiseBuffer;

        // Population size can just be the head number
        public ESNetworkLayer(int populationSize, float noiseStD, int nInputs, int nNeurons, ActivationFunction activationFunction,
            ComputeShader shader,
            bool isFirstLayer = false, float paramsRange = 4, float paramsCoefficient = 0.01f, int headNumber = 1) :
            base(nInputs, nNeurons, activationFunction, shader, isFirstLayer, paramsRange, paramsCoefficient,
                headNumber)
        {
            _weightNoise = new float[nInputs, nNeurons * populationSize];
            _biasNoise = new float[nNeurons * populationSize];

            _weightNoiseBuffer = new ComputeBuffer(_weightNoise.Length, sizeof(float));
            _biasNoiseBuffer = new ComputeBuffer(_biasNoise.Length, sizeof(float));

            _shader.SetInt("noise_row_size", _weightNoise.GetLength(1));
            _shader.SetFloat("noise_std", noiseStD);
            
            SetNoise();
        }

        //TODO: Can store random gaussian values in a huge buffer, and can probably use some the same technique used in
        // noisy nets, although this would not be simple
        public override void Forward(float[,] inputs)
        {
            base.Forward(inputs);
        }

        //TODO: standard backwards pass can cause problems because back input kernel does not exist
        public override void Backward(float[,] dValues, float currentLearningRate, float beta1Corrected,
            float beta2Corrected)
        {
            base.Backward(dValues, currentLearningRate, beta1Corrected, beta2Corrected);

            SetNoise();
        }
        
        private void SetNoise()
        {
            for (int i = 0; i < _weightNoise.GetLength(1); i++)
            {
                _biasNoise[i] = NnMath.RandomGaussian(-4.0f, 4.0f);
                for (int j = 0; j < _weightNoise.GetLength(0); j++)
                {
                    _weightNoise[j, i] = NnMath.RandomGaussian(-4.0f, 4.0f);
                }
            }

            _weightNoiseBuffer.SetData(_weightNoise);
            _biasNoiseBuffer.SetData(_biasNoise);
        }

        protected override void InitializeForwardBuffers(float[,] inputs)
        {
            base.InitializeForwardBuffers(inputs);

            _shader.SetBuffer(_kernelHandleForward, "weight_noise", _weightNoiseBuffer);
            _shader.SetBuffer(_kernelHandleForward, "bias_noise", _biasNoiseBuffer);
        }

        protected override void InitializeBackwardsBuffers(float[,] dValues)
        {
            base.InitializeBackwardsBuffers(dValues);
            
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "weight_noise", _weightNoiseBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "bias_noise", _biasNoiseBuffer);
        }
        
        public override void Dispose()
        {
            base.Dispose();

            _weightNoiseBuffer?.Dispose();
            _biasNoiseBuffer?.Dispose();
        }
    }
}