using Algorithms.RL;
using DL;
using DL.NN;
using UnityEngine;

namespace TestGround
{
    public class TestRainbowDQN : TestDQN
    {
        [SerializeField] private int stepNumber;
        [SerializeField] protected ComputeShader noisyShader;
        [SerializeField] private int supportSize;
        [SerializeField] private float supportMinValue;
        [SerializeField] private int supportMaxValue;
        [SerializeField] private float beta;
        [SerializeField] private float alpha;
        [SerializeField] private float sigma;

        public override string GetDescription()
        {
            return "Rainbow DQN, n-step " + stepNumber + ", max V " + supportMaxValue + " min V " + supportMinValue +
                   ", alpha " + alpha + ", beta " + beta + ", sigma " + sigma + " , 3 layers, " + neuronNumber +
                   " neurons, " + activationFunction + ", " + batchSize + " batch size, " + gamma + " gamma, " +
                   targetNetworkCopyPeriod + "  copy network, lr " + learningRate + ", decay " + decayRate +
                   ", initialization std " + weightsInitStd;
        }

        protected override void Start()
        {
            _currentSate = _env.ResetEnv();

            var inputLayers = new Layer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true),
            };
            var inputModel = new NetworkModel(inputLayers, new NoLoss(Instantiate(shader)), learningRate, decayRate);

            //Value model might not need to have a noisy shader since it does not affect the chosen action
            var valueLayers = new Layer[]
            {
                new NoisyNetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(noisyShader), sigma),
                new NoisyNetworkLayer(neuronNumber, supportSize, ActivationFunction.Linear, Instantiate(noisyShader), sigma),
            };
            var valueModel = new NetworkModel(valueLayers, new NoLoss(Instantiate(shader)), learningRate, decayRate);

            var advantageLayers = new Layer[]
            {
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader)),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions * supportSize, ActivationFunction.Linear,
                    Instantiate(shader))
            };
            var advantageModel =
                new NetworkModel(advantageLayers, new NoLoss(Instantiate(shader)), learningRate, decayRate);

            DuellingNetwork updateModel = new DuellingNetwork(inputModel, valueModel, advantageModel);

            // target creation
            var inputTargetLayers = new Layer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true),
            };
            var inputTargetModel = new NetworkModel(inputTargetLayers, new NoLoss(Instantiate(shader)), learningRate,
                decayRate);

            var valueTargetLayers = new Layer[]
            {
                new NoisyNetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(noisyShader), sigma),
                new NoisyNetworkLayer(neuronNumber, supportSize, ActivationFunction.Linear, Instantiate(noisyShader), sigma),
            };
            var valueTargetModel = new NetworkModel(valueTargetLayers, new NoLoss(Instantiate(shader)));

            var advantageTargetLayers = new Layer[]
            {
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader)),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions * supportSize, ActivationFunction.Linear,
                    Instantiate(shader))
            };
            var advantageTargetModel = new NetworkModel(advantageTargetLayers, new NoLoss(Instantiate(shader)));

            DuellingNetwork targetModel = new DuellingNetwork(inputTargetModel, valueTargetModel, advantageTargetModel);

            // Init rainbow model here
            _DQN = new RainbowDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize,
                stepNumber, supportSize, supportMinValue, supportMaxValue, beta, alpha);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}