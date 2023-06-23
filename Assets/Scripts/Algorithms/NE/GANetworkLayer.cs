using NN;
using NN.CPU_Single;
using UnityEngine;
using ActivationFunction = NN.ActivationFunction;

namespace Algorithms.NE
{
    public struct CrossoverInfo
    {
        public int Parent1;
        public int Parent2;
        public int CrossoverPoint;

        public CrossoverInfo(int parent1, int parent2, int crossoverPoint)
        {
            Parent1 = parent1;
            Parent2 = parent2;
            CrossoverPoint = crossoverPoint;
        }
    }

    public class GANetworkLayer : NetworkLayer
    {
        private readonly float[] _weightsMutationNoise;
        private readonly float[] _biasesMutationNoise;

        private readonly ComputeBuffer _weightsTempBuffer;
        private readonly ComputeBuffer _biasesTempBuffer;
        private readonly ComputeBuffer _weightsMutationNoiseBuffer;
        private readonly ComputeBuffer _biasesMutationNoiseBuffer;
        private readonly ComputeBuffer _crossoverInfoBuffer;

        // Cashed variables
        private readonly int _populationSize;
        private readonly int _neuronNumber;
        private readonly int _individualWeightSize;

        private readonly int[] _populationNoiseIndexes;

        private readonly int _noiseSamplesSize;
        private readonly float[] _noiseSamplesBuffer;


        public GANetworkLayer(int populationSize, int nInputs, int nNeurons, ActivationFunction activationFunction,
            ComputeShader shader, float noiseRange = 10.0f, bool isFirstLayer = true, float paramsRange = 4,
            float paramsCoefficient = 0.005f, int headNumber = 1) : base(nInputs, nNeurons * populationSize,
            activationFunction, shader, isFirstLayer, paramsRange, paramsCoefficient, headNumber)
        {
            _populationSize = populationSize;
            _neuronNumber = nNeurons;
            _individualWeightSize = nInputs * nNeurons;

            _shader.SetInt("weights_row_size", nNeurons);
            _shader.SetInt("population_weight_row_size", nNeurons * populationSize);

            _kernelHandleWeightsBiasesBackward = _shader.FindKernel("GA_backwards_pass");

            _weightsTempBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _biasesTempBuffer = new ComputeBuffer(_biases.Length, sizeof(float));

            _weightsTempBuffer.SetData(_weights);
            _biasesTempBuffer.SetData(_biases);

            _noiseSamplesSize = 1000000;
            _noiseSamplesBuffer = new float[_noiseSamplesSize];
            for (int i = 0; i < _noiseSamplesSize; i++)
            {
                _noiseSamplesBuffer[i] = NnMath.RandomGaussian(-noiseRange, noiseRange);
            }

            _weightsMutationNoise = new float[_weights.Length];
            _weightsMutationNoiseBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _weightsMutationNoiseBuffer.SetData(_weightsMutationNoise);

            _biasesMutationNoise = new float[_biases.Length];
            _biasesMutationNoiseBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _biasesMutationNoiseBuffer.SetData(_biasesMutationNoise);

            _populationNoiseIndexes = new int[_weights.Length];

            _crossoverInfoBuffer = new ComputeBuffer(populationSize, sizeof(int) * 3);

            InitializeBackwardsBuffers(new float[1, 1]);
        }

        public void UpdateLayer(CrossoverInfo[] crossoverInfos, float[] mutationsVolume)
        {
            //TODO: crossover point must be decided here
            var totalMutations = 0;
            for (int i = 0; i < _populationSize; i++)
            {
                var noiseIndexStart = totalMutations;
                var mutationVolume = (int)(mutationsVolume[i] * _individualWeightSize);
                totalMutations += mutationVolume;

                var rangeMin = _individualWeightSize * i;
                var rangeMax = _individualWeightSize * (i + 1);
                crossoverInfos[i].CrossoverPoint = Random.Range(rangeMin, rangeMax);
                
                for (int j = 0; j < mutationVolume; j++)
                {
                    var randomIndex = Random.Range(rangeMin, rangeMax);
                    _populationNoiseIndexes[noiseIndexStart + j] = randomIndex;
                    _weightsMutationNoise[randomIndex] = _noiseSamplesBuffer[Random.Range(0, _noiseSamplesSize)];
                }
            }

            _weightsMutationNoiseBuffer.SetData(_weightsMutationNoise);
            _crossoverInfoBuffer.SetData(crossoverInfos);

            //TODO: must be protected
            // _shader.Dispatch(_kernelHandleInputsBackward, _threadGroupXInputsBackward,
            //     _threadGroupYInputsBackward, 1);

            for (int i = 0; i < totalMutations; i++)
            {
                _weightsMutationNoise[_populationNoiseIndexes[i]] = 0f;
            }
        }

        protected override void InitializeBackwardsBuffers(float[,] dValues)
        {
            base.InitializeBackwardsBuffers(dValues);

            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "weights_temp", _weightsTempBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "biases_temp", _biasesTempBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "weights_mutation_noise",
                _weightsMutationNoiseBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "biases_mutation_noise", _weightsMutationNoiseBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "crossover_info", _crossoverInfoBuffer);
        }

        public override void Dispose()
        {
            _weightsTempBuffer?.Dispose();
            _biasesTempBuffer?.Dispose();
            _weightsMutationNoiseBuffer?.Dispose();
            _biasesMutationNoiseBuffer?.Dispose();
            _crossoverInfoBuffer?.Dispose();
        }
    }
}