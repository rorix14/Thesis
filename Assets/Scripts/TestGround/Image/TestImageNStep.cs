using Algorithms.RL;
using DL;
using DL.CNN;
using DL.NN;
using Gym;
using UnityEngine;

namespace TestGround.Image
{
    public class TestImageNStep : TestNStepDQN
    {
        [SerializeField] private ComputeShader shaderCNN;

        public override string GetDescription()
        {
            return "Description not yet done";
        }

        protected override void Start()
        {
            _env = FindObjectOfType<ImageStealthGameEnv>();
            var envImage = (ImageStealthGameEnv)_env;

            _currentSate = _env.ResetEnv();

            var inputDepth = envImage.IsGrayscale ? 1 : 3;

            var updateLayers = new Layer[]
            {
                // model V1
                // new ConvolutionalLayer(envImage.ImageWithHeight, inputDepth, 5, 4, 1, true, Instantiate(shaderCNN),
                //     true),
                // new ConvolutionalLayer(12, 4, 3, 8, 1, true, Instantiate(shaderCNN)),
                // new NetworkLayer(5 * 5 * 8, neuronNumber, activationFunction, Instantiate(shader),
                //     paramsCoefficient: weightsInitStd),
                // model V2
                // new ConvolutionalLayer(envImage.ImageWithHeight, inputDepth, 3, 16, 1, true, Instantiate(shaderCNN),
                // true),
                // new ConvolutionalLayer(13, 16, 3, 32, 1, true, Instantiate(shaderCNN)),
                // new NetworkLayer(5 * 5 * 32, neuronNumber, activationFunction, Instantiate(shader),
                //     paramsCoefficient: weightsInitStd),
                // model V3
                new ConvolutionalLayer(envImage.ImageWithHeight, inputDepth, 3, 16, 2, false, Instantiate(shaderCNN),
                    true),
                new ConvolutionalLayer(13, 16, 3, 32, 1, false, Instantiate(shaderCNN)),
                new NetworkLayer(11 * 11 * 32, neuronNumber, activationFunction, Instantiate(shader),
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear,
                    Instantiate(shader), paramsCoefficient: weightsInitStd)
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            var targetLayers = new Layer[]
            {
                // new ConvolutionalLayer(envImage.ImageWithHeight, inputDepth, 5, 4, 1, true, Instantiate(shaderCNN),
                //     true),
                // new ConvolutionalLayer(12, 4, 3, 8, 1, true, Instantiate(shaderCNN)),
                // new NetworkLayer(5 * 5 * 8, neuronNumber, activationFunction, Instantiate(shader),
                //     paramsCoefficient: weightsInitStd),
                // new ConvolutionalLayer(envImage.ImageWithHeight, inputDepth, 3, 16, 1, true, Instantiate(shaderCNN),
                //     true),
                // new ConvolutionalLayer(13, 16, 3, 32, 1, true, Instantiate(shaderCNN)),
                // new NetworkLayer(5 * 5 * 32, neuronNumber, activationFunction, Instantiate(shader),
                //     paramsCoefficient: weightsInitStd),
                new ConvolutionalLayer(envImage.ImageWithHeight, inputDepth, 3, 16, 2, false, Instantiate(shaderCNN),
                    true),
                new ConvolutionalLayer(13, 16, 3, 32, 1, false, Instantiate(shaderCNN)),
                new NetworkLayer(11 * 11 * 32, neuronNumber, activationFunction, Instantiate(shader),
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear,
                    Instantiate(shader), paramsCoefficient: weightsInitStd)
            };
            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            _DQN = new ModelNStepDQN(stepNumber, updateModel, targetModel, _env.GetNumberOfActions,
                _env.GetObservationSize, batchSize: batchSize, gamma: gamma);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }
    }
}