using System;
using System.Collections;
using System.Collections.Generic;
using Algorithms.RL;
using Graphs;
using Gym;
using NN;
using TestGround.Base;
using UnityEngine;
using UnityEngine.UI;

namespace TestGround
{
    public class TestDQN : TestAlgorithmBase
    {
        [SerializeField] protected ComputeShader shader;
        [SerializeField] private WindowGraph windowGraphPrefab;
        [SerializeField] protected float simulationSpeed;
        [SerializeField] private int numberOfEpisodes;
        [SerializeField] protected int targetNetworkCopyPeriod;
        private int _episodeIndex;
        protected float[] _currentSate;
        private int _totalIteration;
        protected StealthGameEnv _env;

        // DQN specific variables
        protected ModelDQN _DQN;
        protected float _epsilon;

        private WindowGraph _graphReward;
        private WindowGraph _graphLoss;

        // private Stopwatch _stopwatch;
        // private List<long> _times;

        [SerializeField] protected int batchSize;
        [SerializeField] protected float gamma;

        //Only were for testing
        [SerializeField] protected ActivationFunction activationFunction;
        [SerializeField] protected float learningRate;
        [SerializeField] protected float decayRate;
        [SerializeField] protected int neuronNumber;
        [SerializeField] protected float weightsInitStd;

        public override string GetDescription()
        {
            return "DQN, 3 layers, " + neuronNumber + " neurons, " + activationFunction +
                   ", " + batchSize + " batch size, " + gamma + " gamma, " + targetNetworkCopyPeriod +
                   "  copy network, lr " + learningRate + ", decay " + decayRate + ", initialization std " +
                   weightsInitStd;
        }

        private void Awake()
        {
            _env = FindObjectOfType<StealthGameEnv>();
            Rewards = new List<float>(numberOfEpisodes);
            Loss = new List<float>(numberOfEpisodes);
            for (int i = 0; i < Rewards.Capacity; i++)
            {
                Rewards.Add(0f);
                Loss.Add(0f);
            }

            // _stopwatch = new Stopwatch();
            // _times = new List<long>(1000000);
        }

        protected virtual void Start()
        {
            _currentSate = _env.ResetEnv();

            var updateLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true,
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader),
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader),
                    paramsCoefficient: weightsInitStd)
            };
            var updateModel = new NetworkModel(updateLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            var targetLayers = new NetworkLayer[]
            {
                new NetworkLayer(_env.GetObservationSize, neuronNumber, activationFunction, Instantiate(shader), true,
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, neuronNumber, activationFunction, Instantiate(shader),
                    paramsCoefficient: weightsInitStd),
                new NetworkLayer(neuronNumber, _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader),
                    paramsCoefficient: weightsInitStd)
            };
            var targetModel = new NetworkModel(targetLayers, new MeanSquaredError(Instantiate(shader)), learningRate,
                decayRate);

            _DQN = new ModelDQN(updateModel, targetModel, _env.GetNumberOfActions, _env.GetObservationSize,
                batchSize: batchSize, gamma: gamma);

            _DQN.SetTargetModel();

            _epsilon = 1.0f;
            Time.timeScale = simulationSpeed;
        }

        private void FixedUpdate()
        {
            if (_episodeIndex >= numberOfEpisodes)
            {
                //if (!_env) return;

                IsFinished = true;
                //_env.Close();
                //PlotTrainingData();
                return;
            }

            int action = _DQN.EpsilonGreedySample(_currentSate, _epsilon);
            var stepInfo = _env.Step(action);

            _DQN.AddExperience(_currentSate, action, stepInfo.Reward, stepInfo.Done, stepInfo.Observation);

            //_stopwatch.Restart();
            _DQN.Train();
            //_stopwatch.Stop();
            //_times.Add(_stopwatch.ElapsedMilliseconds);

            if (_totalIteration % targetNetworkCopyPeriod == 0)
            {
                _DQN.SetTargetModel();
            }

            _currentSate = stepInfo.Observation;
            Rewards[_episodeIndex] += stepInfo.Reward;
            ++_totalIteration;

            if (!stepInfo.Done) return;

            Loss[_episodeIndex] = _DQN.SampleLoss();
            _currentSate = _env.ResetEnv();
            ++_episodeIndex;

            // could clamp epsilon value with a max function in order to have a min exploration value
            _epsilon = 1.0f / (float)Math.Sqrt(_episodeIndex + 1);
            //_epsilon = 0.0f;
        }

        private void PlotTrainingData()
        {
            Time.timeScale = 1;

            float rewardSum = 0.0f;
            foreach (var reward in Rewards)
            {
                rewardSum += reward;
            }

            float lossSum = 0.0f;
            for (int i = 0; i < Loss.Count; i++)
            {
                var loss = Loss[i];
                lossSum += loss;
                // if (loss > 1)
                // {
                //     _lossPerEpisode[i] = 1.0f;
                // }
            }

            // float timeSum = 0.0f;
            // foreach (var time in _times)
            // {
            //     timeSum += time;
            // }


            print("Average Reward: " + rewardSum / Rewards.Count);
            print("Average Loss: " + lossSum / Loss.Count);
            //print("Average Time: " + timeSum / _times.Count);

            var layoutGroup = FindObjectOfType<VerticalLayoutGroup>();
            _graphReward = Instantiate(windowGraphPrefab, layoutGroup.transform);
            _graphLoss = Instantiate(windowGraphPrefab, layoutGroup.transform);
            StartCoroutine(ShowGraphs());
        }

        private IEnumerator ShowGraphs()
        {
            yield return new WaitForSeconds(0.1f);

            _graphReward.SetGraph(null, Rewards, GraphType.LineGraph,
                "Rewards per Episode", "episodes", "rewards");

            _graphLoss.SetGraph(null, Loss, GraphType.LineGraph,
                "Loss per Episode", "episodes", "loss");
        }

        private void OnDestroy()
        {
            //Time.timeScale = 1;
            _DQN?.Dispose();
        }
    }
}