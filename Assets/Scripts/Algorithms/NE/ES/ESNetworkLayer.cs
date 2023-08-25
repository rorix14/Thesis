using NN;
using NN.CPU_Single;
using UnityEngine;
using ActivationFunction = NN.ActivationFunction;

namespace Algorithms.NE
{
    public enum AlgorithmNE
    {
        RS,
        ES
    }

    public class ESNetworkLayer : NetworkLayer
    {
        private readonly float[] _noiseInputOutput;
        private readonly ComputeBuffer _noiseInputOutputBuffer;

        private readonly float[] _noiseSamplesBuffer;

        // Cashed variables
        private readonly int _noiseInputOutputLenght;

        private readonly int _noiseSamplesSize;
        
        private readonly int _bestPerformerIndexID;
        private readonly int _rewardMeanID;
        private readonly int _rewardStdID;
        
        public ESNetworkLayer(AlgorithmNE algorithmNE, int populationSize, float noiseStD, int nInputs, int nNeurons,
            ActivationFunction activationFunction, ComputeShader shader, float noiseRange = 10.0f,
            bool isFirstLayer = true, float paramsRange = 4.0f, float paramsCoefficient = 0.01f, int headNumber = 1) :
            base(nInputs, nNeurons, activationFunction, shader, isFirstLayer, paramsRange, paramsCoefficient,
                headNumber)
        {
            var noiseRowSize = nNeurons * (populationSize / 2);
            _shader.SetInt("noise_row_size", noiseRowSize);
            _shader.SetFloat("noise_std", noiseStD);
            _shader.SetFloat("noise_normalizer", noiseStD * populationSize);
            _shader.SetInt("half_population_size", populationSize / 2);

            switch (algorithmNE)
            {
                case AlgorithmNE.RS:
                    _kernelHandleWeightsBiasesBackward = _shader.FindKernel("RS_backwards_pass");
                    break;
                case AlgorithmNE.ES:
                    _kernelHandleWeightsBiasesBackward = _shader.FindKernel("ES_backwards_pass");
                    break;
            }

            _noiseSamplesSize = 10000000;
            _noiseSamplesBuffer = new float[_noiseSamplesSize];
            for (int i = 0; i < _noiseSamplesSize; i++)
            {
                var randomValue = NnMath.RandomGaussian(-noiseRange, noiseRange);
                _noiseSamplesBuffer[i] = NnMath.Sign(randomValue) * Mathf.Sqrt(Mathf.Abs(randomValue));
            }

            _noiseInputOutput = new float[nInputs + noiseRowSize];
            _noiseInputOutputLenght = _noiseInputOutput.Length;

            _noiseInputOutputBuffer = new ComputeBuffer(_noiseInputOutputLenght, sizeof(float));

            SetNoise();
            
            _bestPerformerIndexID = Shader.PropertyToID("best_performer_index");
            _rewardMeanID = Shader.PropertyToID("reward_mean");
            _rewardStdID = Shader.PropertyToID("reward_std");
        }

        public void SetNoiseStd(float noiseStd)
        {
            _shader.SetFloat("noise_std", noiseStd);
        }

        public void SetBestIndex(int bestIndex)
        {
            _shader.SetInt(_bestPerformerIndexID, bestIndex);
        }

        public void SetNeParameters(float rewardMean, float rewardStd)
        {
            _shader.SetFloat(_rewardMeanID, rewardMean);
            _shader.SetFloat(_rewardStdID, rewardStd);
        }

        public override void Backward(float[,] dValues, float currentLearningRate, float beta1Corrected,
            float beta2Corrected)
        {
            base.Backward(dValues, currentLearningRate, beta1Corrected, beta2Corrected);

            SetNoise();
        }

        public void SetNoise()
        {
            for (int i = 0; i < _noiseInputOutputLenght; i++)
            {
                _noiseInputOutput[i] = _noiseSamplesBuffer[Random.Range(0, _noiseSamplesSize)];
            }

            _noiseInputOutputBuffer.SetData(_noiseInputOutput);
        }

        protected override void InitializeForwardBuffers(float[,] inputs)
        {
            base.InitializeForwardBuffers(inputs);

            _shader.SetBuffer(_kernelHandleForward, "noise_input_output_buffer", _noiseInputOutputBuffer);
        }

        protected override void InitializeBackwardsBuffers(float[,] dValues)
        {
            base.InitializeBackwardsBuffers(dValues);

            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "noise_input_output_buffer", _noiseInputOutputBuffer);
        }

        public override void Dispose()
        {
            base.Dispose();

            _noiseInputOutputBuffer?.Dispose();
        }
    }
}