using DL.NN;
using NN;
using NN.CPU_Single;
using UnityEngine;
using ActivationFunction = DL.NN.ActivationFunction;

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
        private readonly int _inputNumber;
        private readonly int _individualWeightSize;
        private readonly int _populationNeuronLenght;

        private readonly int[] _populationWeightNoiseIndexes;
        private readonly int[] _populationBiasesNoiseIndexes;

        private readonly int _noiseSamplesSize;
        private readonly float[] _noiseSamplesBuffer;

        // private readonly float[,] _weigthTest;
        // private readonly float[,] _biasTest;

        public GANetworkLayer(int populationSize, float noiseStD, int nInputs, int nNeurons,
            ActivationFunction activationFunction, ComputeShader shader, float noiseRange = 10.0f,
            float paramsRange = 4, float paramsCoefficient = 0.01f) : base(nInputs, nNeurons * populationSize,
            activationFunction, shader, true, paramsRange, paramsCoefficient)
        {
            _populationSize = populationSize;
            _neuronNumber = nNeurons;
            _inputNumber = nInputs;
            _individualWeightSize = nInputs * nNeurons;
            _populationNeuronLenght = populationSize * nNeurons;

            _shader.SetInt("weights_row_size", nNeurons);
            _shader.SetInt("population_weight_row_size", nNeurons * populationSize);

            _kernelHandleWeightsBiasesBackward = _shader.FindKernel("GA_backwards_pass_full");
            _kernelHandleInputsBackward = _shader.FindKernel("GA_set_new_population_full");

            _weightsTempBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _biasesTempBuffer = new ComputeBuffer(_biases.Length, sizeof(float));

            // _weigthTest = new float[nInputs, nNeurons * populationSize];
            // _biasTest = new float[1, nNeurons * populationSize];

            _weightsTempBuffer.SetData(_weights);
            _biasesTempBuffer.SetData(_biases);

            _noiseSamplesSize = 1000000;
            _noiseSamplesBuffer = new float[_noiseSamplesSize];
            for (int i = 0; i < _noiseSamplesSize; i++)
            {
                _noiseSamplesBuffer[i] = noiseStD * NnMath.RandomGaussian(-noiseRange, noiseRange);
            }

            _weightsMutationNoise = new float[_weights.Length];
            _weightsMutationNoiseBuffer = new ComputeBuffer(_weights.Length, sizeof(float));
            _weightsMutationNoiseBuffer.SetData(_weightsMutationNoise);
            _populationWeightNoiseIndexes = new int[_weights.Length];

            _biasesMutationNoise = new float[_biases.Length];
            _biasesMutationNoiseBuffer = new ComputeBuffer(_biases.Length, sizeof(float));
            _biasesMutationNoiseBuffer.SetData(_biasesMutationNoise);
            _populationBiasesNoiseIndexes = new int[_biases.Length];

            _crossoverInfoBuffer = new ComputeBuffer(populationSize, sizeof(int) * 3);

            InitializeBuffers(new float[populationSize, nInputs]);
        }

        public void UpdateLayer(CrossoverInfo[] crossoverInfos, float[] mutationsVolume)
        {
            var totalWeightsMutations = 0;
            var totalBiasMutations = 0;
            var eliteNumber = 0;
            for (int i = 0; i < _populationSize; i++)
            {
                if (crossoverInfos[i].Parent1 == crossoverInfos[i].Parent2)
                {
                    eliteNumber++;
                    continue;
                }

                //crossoverInfos[i].CrossoverPoint = Random.Range(0, _neuronNumber);
                crossoverInfos[i].CrossoverPoint = (i - eliteNumber) % 2 == 0
                    ? Random.Range(0, _neuronNumber)
                    : crossoverInfos[i - 1].CrossoverPoint;

                var mutationVolume = mutationsVolume[i];

                var weightNoiseIndexStart = totalWeightsMutations;
                var weightsMutationVolume = (int)(mutationVolume * _individualWeightSize);
                totalWeightsMutations += weightsMutationVolume;

                var rangeMin = _neuronNumber * i;
                var rangeMax = _neuronNumber * (i + 1);

                for (int j = 0; j < weightsMutationVolume; j++)
                {
                    var randomIndex = Random.Range(0, _inputNumber) * _populationNeuronLenght +
                                      Random.Range(rangeMin, rangeMax);

                    _weightsMutationNoise[randomIndex] = _noiseSamplesBuffer[Random.Range(0, _noiseSamplesSize)];
                    _populationWeightNoiseIndexes[weightNoiseIndexStart + j] = randomIndex;
                }

                var biasNoiseIndexStart = totalBiasMutations;
                var biasMutationVolume = (int)(mutationVolume * _neuronNumber);
                totalBiasMutations += biasMutationVolume;

                rangeMin = _neuronNumber * i;
                rangeMax = _neuronNumber * (i + 1);

                for (int j = 0; j < biasMutationVolume; j++)
                {
                    var randomIndex = Random.Range(rangeMin, rangeMax);
                    _biasesMutationNoise[randomIndex] = _noiseSamplesBuffer[Random.Range(0, _noiseSamplesSize)];
                    _populationBiasesNoiseIndexes[biasNoiseIndexStart + j] = randomIndex;
                }
            }

            _weightsMutationNoiseBuffer.SetData(_weightsMutationNoise);
            _biasesMutationNoiseBuffer.SetData(_biasesMutationNoise);
            _crossoverInfoBuffer.SetData(crossoverInfos);
            
            _shader.Dispatch(_kernelHandleWeightsBiasesBackward, _threadGroupXInputsBackward,
                _threadGroupYInputsBackward, _threadGroupZInputsBackward);
            _shader.Dispatch(_kernelHandleInputsBackward, _threadGroupXInputsBackward, _threadGroupYInputsBackward,
                _threadGroupZInputsBackward);

            // var testW = new float[_weights.GetLength(0), _weights.GetLength(1)];
            // var testB = new float[_biases.GetLength(0), _biases.GetLength(1)];
            // _weightsTempBuffer.GetData(testW);
            // _biasesTempBuffer.GetData(testB);
            
            for (int i = 0; i < totalWeightsMutations; i++)
            {
                _weightsMutationNoise[_populationWeightNoiseIndexes[i]] = 0f;
            }

            for (int i = 0; i < totalBiasMutations; i++)
            {
                _biasesMutationNoise[_populationBiasesNoiseIndexes[i]] = 0f;
            }
        }

        private int _threadGroupZInputsBackward;

        private void InitializeBuffers(float[,] inputShape)
        {
            Output = new float[inputShape.GetLength(0), _neuronNumber];
            InitializeForwardBuffers(inputShape);
            InitializeBackwardsBuffers(new float[1, 1]);

            _shader.GetKernelThreadGroupSizes(_kernelHandleInputsBackward, out var x, out var y, out var z);
            _threadGroupXInputsBackward = Mathf.CeilToInt(_weights.GetLength(0) / (float)x);
            _threadGroupYInputsBackward = Mathf.CeilToInt(_neuronNumber / (float)y);
            _threadGroupZInputsBackward = Mathf.CeilToInt(_populationSize / (float)z);

            _shader.SetBuffer(_kernelHandleInputsBackward, "biases", _biasesBuffer);
            _shader.SetBuffer(_kernelHandleInputsBackward, "weights_temp", _weightsTempBuffer);
            _shader.SetBuffer(_kernelHandleInputsBackward, "biases_temp", _biasesTempBuffer);

            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "weights_temp", _weightsTempBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "biases_temp", _biasesTempBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "weights_mutation_noise",
                _weightsMutationNoiseBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "biases_mutation_noise", _biasesMutationNoiseBuffer);
            _shader.SetBuffer(_kernelHandleWeightsBiasesBackward, "crossover_info", _crossoverInfoBuffer);
        }

        public override void Dispose()
        {
            base.Dispose();

            _weightsTempBuffer?.Dispose();
            _biasesTempBuffer?.Dispose();
            _weightsMutationNoiseBuffer?.Dispose();
            _biasesMutationNoiseBuffer?.Dispose();
            _crossoverInfoBuffer?.Dispose();
        }
    }
}