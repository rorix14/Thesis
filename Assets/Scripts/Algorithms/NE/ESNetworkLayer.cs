using NN;
using NN.CPU_Single;
using UnityEngine;
using ActivationFunction = NN.ActivationFunction;
using Random = UnityEngine.Random;

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

        private readonly float[] _inputNoise;
        private readonly float[] _outputNoise;

        // Cashed variables
        private readonly int _inputSize;
        private readonly int _noiseRowSize;

        private readonly int _noiseSamplesSize;
        private readonly float[] _noiseSamplesBuffer;
        
        // Population size can just be the head number
        public ESNetworkLayer(int populationSize, float noiseStD, int nInputs, int nNeurons,
            ActivationFunction activationFunction,
            ComputeShader shader,
            bool isFirstLayer = false, float paramsRange = 4, float paramsCoefficient = 0.01f, int headNumber = 1) :
            base(nInputs, nNeurons, activationFunction, shader, isFirstLayer, paramsRange, paramsCoefficient,
                headNumber)
        {
            _inputSize = nInputs;
            _noiseRowSize = nNeurons * (populationSize / 2);

            _weightNoise = new float[nInputs, nNeurons * (populationSize / 2)];
            _biasNoise = new float[nNeurons * (populationSize / 2)];

            //_weightNoiseBuffer = new ComputeBuffer(_weightNoise.Length, sizeof(float));
            //_biasNoiseBuffer = new ComputeBuffer(_biasNoise.Length, sizeof(float));

            _shader.SetInt("noise_row_size", _weightNoise.GetLength(1));
            _shader.SetFloat("noise_std", noiseStD);
            _shader.SetFloat("noise_normalizer", noiseStD * populationSize);
            _shader.SetInt("half_population_size", populationSize / 2);

            _noiseSamplesSize = 10000000;
            _noiseSamplesBuffer = new float[_noiseSamplesSize];
            for (int i = 0; i < _noiseSamplesSize; i++)
            {
                //_noiseSamplesBuffer[i] = noiseStD * NnMath.RandomGaussian(-10.0f, 10.0f);
                _noiseSamplesBuffer[i] = NnMath.RandomGaussian(-10.0f, 10.0f);
            }

            _inputNoise = new float[nInputs];
            _outputNoise = new float[_noiseRowSize];
            
            //TODO: temp names
            _weightNoiseBuffer = new ComputeBuffer(_inputNoise.Length, sizeof(float));
            _biasNoiseBuffer = new ComputeBuffer(_outputNoise.Length, sizeof(float));

            SetNoise();
        }

        public void SetNoiseStd(float noiseStd)
        {
            _shader.SetFloat("noise_std", noiseStd);
        }

        public void SetNeParameters(float rewardMean, float rewardStd)
        {
            _shader.SetFloat("reward_mean", rewardMean);
            _shader.SetFloat("reward_std", rewardStd);
        }

        //TODO: Can store random gaussian values in a huge buffer, and can probably use some the same technique used in
        // noisy nets, although this would not be simple

        public override void Backward(float[,] dValues, float currentLearningRate, float beta1Corrected,
            float beta2Corrected)
        {
            base.Backward(dValues, currentLearningRate, beta1Corrected, beta2Corrected);
            
            SetNoise();
        }

        private void SetNoise()
        {
            // for (int i = 0; i < _noiseRowSize; i++)
            // {
            //     _biasNoise[i] = _noiseSamplesBuffer[Random.Range(0, _noiseSamplesSize)];
            //     for (int j = 0; j < _inputSize; j++)
            //     {
            //         _weightNoise[j, i] = _noiseSamplesBuffer[Random.Range(0, _noiseSamplesSize)];
            //     }
            // }

            for (int i = 0; i < _inputSize; i++)
            {
                _inputNoise[i] = _noiseSamplesBuffer[Random.Range(0, _noiseSamplesSize)];
            }

            for (int i = 0; i < _noiseRowSize; i++)
            {
                _outputNoise[i] = _noiseSamplesBuffer[Random.Range(0, _noiseSamplesSize)];
            }

            _weightNoiseBuffer.SetData(_inputNoise);
            _biasNoiseBuffer.SetData(_outputNoise);
        }

        protected override void InitializeForwardBuffers(float[,] inputs)
        {
            base.InitializeForwardBuffers(inputs);

            //_shader.SetBuffer(_kernelHandleForward, "weight_noise", _weightNoiseBuffer);
            //_shader.SetBuffer(_kernelHandleForward, "bias_noise", _biasNoiseBuffer);
            
            _shader.SetBuffer(_kernelHandleForward, "input_noise", _weightNoiseBuffer);
            _shader.SetBuffer(_kernelHandleForward, "output_noise", _biasNoiseBuffer);
        }

        protected override void InitializeBackwardsBuffers(float[,] dValues)
        {
            base.InitializeBackwardsBuffers(dValues);

            //_shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "weight_noise", _weightNoiseBuffer);
            //_shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "bias_noise", _biasNoiseBuffer);
            
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "input_noise", _weightNoiseBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "output_noise", _biasNoiseBuffer);
        }

        public override void Dispose()
        {
            base.Dispose();

            _weightNoiseBuffer?.Dispose();
            _biasNoiseBuffer?.Dispose();
        }
    }
}