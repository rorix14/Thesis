using NN;
using NN.CPU_Single;
using UnityEngine;
using ActivationFunction = NN.ActivationFunction;

namespace Algorithms.NE
{
    public class ES : NetworkLayer
    {
        private float _noiseStD;
        private float _learningRate;
        private readonly int _populationSize;
        
        private readonly float[,] _weightNoise;
        private readonly float[] _biasNoise;
        private readonly float _totalNoiseValues;
        
        private readonly ComputeBuffer _weightNoiseBuffer;
        private readonly ComputeBuffer _biasNoiseBuffer;


        // Population size can just be the head number
        public ES(int populationSize, int nInputs, int nNeurons, ActivationFunction activationFunction, ComputeShader shader,
            bool isFirstLayer = false, float paramsRange = 4, float paramsCoefficient = 0.01f, int headNumber = 1) :
            base(nInputs, nNeurons, activationFunction, shader, isFirstLayer, paramsRange, paramsCoefficient,
                headNumber)
        {
            _populationSize = populationSize;

            _weightNoise = new float[nInputs, nNeurons * populationSize];
            _biasNoise = new float[nNeurons * populationSize];
            _totalNoiseValues = _weightNoise.Length;

            _weightNoiseBuffer = new ComputeBuffer(_weightNoise.Length, sizeof(float));
            _biasNoiseBuffer = new ComputeBuffer(_biasNoise.Length, sizeof(float));
            
            _shader.SetInt("noise_row_size", _weightNoise.GetLength(1));
        }

        public override void Forward(float[,] inputs)
        {
            for (int i = 0; i < _weightNoise.GetLength(1); i++)
            {
                _biasNoise[i] =  NnMath.RandomGaussian(-4.0f, 4.0f);
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
        public override void Dispose()
        {
            base.Dispose();
            
            _weightNoiseBuffer?.Dispose();
        }
    }
}