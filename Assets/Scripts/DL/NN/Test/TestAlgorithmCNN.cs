using Algorithms.RL;
using DL.CNN;
using Gym;
using UnityEngine;

namespace DL.NN.Test
{
    public class TestAlgorithmCNN : MonoBehaviour
    {
        [SerializeField] private ComputeShader shaderCNN;
        [SerializeField] protected ComputeShader shader;

        [SerializeField] protected float simulationSpeed;

        [SerializeField] protected ActivationFunction activationFunction;
        [SerializeField] protected float learningRate;
        [SerializeField] protected float decayRate;
        [SerializeField] protected int neuronNumber;
        [SerializeField] protected float weightsInitStd;

        [SerializeField] protected int batchSize;
        [SerializeField] protected float gamma;
        [SerializeField] protected int targetNetworkCopyPeriod;

        protected ModelDQN _DQN;

        protected float[] _currentSate;
        protected StealthGameEnv _env;

        [SerializeField] private int totalIteration;

        //private bool _startRecording;

        private void Awake()
        {
            _env = FindObjectOfType<ImageStealthGameEnv>();
        }

        void Start()
        {
            _currentSate = _env.ResetEnv();

            //Assuming square images
            var imageWidthHeight = (int)Mathf.Sqrt(_env.GetObservationSize);

            var updateLayers = new Layer[]
            {
                new ConvolutionalLayer(imageWidthHeight, 1, 5, 4, 1, true,
                    Instantiate(shaderCNN), true),
                new ConvolutionalLayer(12, 4, 3, 8, 1, true, Instantiate(shaderCNN)),
                new NetworkLayer(5 * 5 * 8, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader),
                    paramsCoefficient: weightsInitStd)
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            var targetLayers = new Layer[]
            {
                new ConvolutionalLayer(imageWidthHeight, 1, 5, 4, 1, true,
                    Instantiate(shaderCNN), true),
                new ConvolutionalLayer(12, 4, 3, 8, 1, true, Instantiate(shaderCNN)),
                new NetworkLayer(5 * 5 * 8, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader),
                    paramsCoefficient: weightsInitStd)
            };
            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            _DQN = new ModelDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize,
                batchSize: batchSize, gamma: gamma);

            _DQN.SetTargetModel();

            Time.timeScale = simulationSpeed;
        }

        private void FixedUpdate()
        {
            _DQN.EpsilonGreedySample(_currentSate, 0);

            var stepInfo = _env.Step(0);

            _DQN.AddExperience(_currentSate, 0, 0, stepInfo.Done, stepInfo.Observation);

            _DQN.Train();

            _currentSate = stepInfo.Observation;

            if (totalIteration % targetNetworkCopyPeriod == 0)
            {
                _DQN.SetTargetModel();
            }

            //if (totalIteration > 100) _startRecording = true;

            ++totalIteration;
        }

        private void OnDestroy()
        {
            //Time.timeScale = 1;
            _DQN?.Dispose();
        }
    }
}