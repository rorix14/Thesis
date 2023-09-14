using System.Collections;
using System.Collections.Generic;
using Algorithms.NE;
using Graphs;
using Gym;
using NN;
using TestGround.Base;
using UnityEngine;
using UnityEngine.UI;

namespace TestGround.NE
{
    public class TestES : TestAlgorithmBase
    {
        [SerializeField] protected int populationSize;
        [SerializeField] protected float noiseStandardDeviation;
        [SerializeField] protected int requiredSimulations;
        [SerializeField] protected ComputeShader shader;
        [SerializeField] private WindowGraph windowGraphPrefab;
        [SerializeField] protected float simulationSpeed;
        [SerializeField] private int numberOfEpisodes;

        private int _episodeIndex;
        protected JobStealthGameEnv _env;
        protected ES _neModel;

        protected float[,] _currentSates;
        
        private WindowGraph _graphReward;
        private WindowGraph _graphBestIndividualReward;
        
        //Only were for testing
        [SerializeField] protected ActivationFunction activationFunction;
        [SerializeField] protected float learningRate;
        [SerializeField] protected float decayRate;
        [SerializeField] protected int neuronNumber;
        [SerializeField] protected float weightsInitStd;

        public override string GetDescription()
        {
            return "DQN, 2 layers, " + neuronNumber + " neurons, " + activationFunction +
                   ", " + populationSize + " population size, noise std " + noiseStandardDeviation + ", lr " +
                   learningRate + ", decay " + decayRate + ", initialization std " + weightsInitStd;
        }

        private void Awake()
        {
            _env = FindObjectOfType<JobStealthGameEnv>();
            
            Rewards = new List<float>(numberOfEpisodes);
            Loss = new List<float>(numberOfEpisodes);
            for (int i = 0; i < Rewards.Capacity; i++)
            {
                Rewards.Add(0f);
                Loss.Add(0f);
            }
            
            if (populationSize % 2 != 0)
            {
                populationSize++;
            }
        }

        protected virtual void Start()
        {
            //Random.InitState(42);

            _env.CreatePopulation(populationSize);
            _currentSates = _env.DistributedResetEnv();

            var network = new NetworkLayer[]
            {
                new ESNetworkLayer(AlgorithmNE.ES, populationSize, noiseStandardDeviation, _env.GetObservationSize,
                    neuronNumber, activationFunction, Instantiate(shader), paramsCoefficient: weightsInitStd),
                // new ESNetworkLayer(AlgorithmNE.ES, populationSize, noiseStandardDeviation, neuronNumber, neuronNumber,
                //     activationFunction, Instantiate(shader), paramsCoefficient: weightsInitStd),
                new ESNetworkLayer(AlgorithmNE.ES, populationSize, noiseStandardDeviation, neuronNumber,
                    _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader),
                    paramsCoefficient: weightsInitStd)
            };

            var neModel = new ESModel(network, learningRate, decayRate);
            _neModel = new ES(neModel, _env.GetNumberOfActions, populationSize);

            Time.timeScale = simulationSpeed;
        }

        protected void FixedUpdate()
        {
            if (_episodeIndex >= numberOfEpisodes)
            {
                IsFinished = true;
                if (!_env) return;
                _env.Close();
                PlotTrainingData();
                return;
            }

            var actions = _neModel.SamplePopulationActions(_currentSates);
            var stepInfo = _env.DistributedStep(actions);

            _neModel.AddExperience(stepInfo.Rewards, stepInfo.Dones);
            _currentSates = stepInfo.Observations;

            if (_neModel.FinishedIndividuals < populationSize) return;

            //_gaModel.DoNoveltySearch(_env.GetPlayersPositions());
            _neModel.Train();
            
            Rewards[_episodeIndex] = _neModel.EpisodeRewardMean;
            Loss[_episodeIndex] = _neModel.EpisodeBestReward;

            _currentSates = _env.DistributedResetEnv();
            ++_episodeIndex;

            // if (_episodeIndex % requiredSimulations == 0)
            // {
            //     _neModel.Train();
            // }
            // else
            // {
            //     _neModel.SoftReset();
            // }
        }

        private void PlotTrainingData()
        {
            Time.timeScale = 1;

            float rewardSum = 0.0f;
            foreach (var reward in Rewards)
            {
                rewardSum += reward;
            }

            float bestIndividualSum = 0.0f;
            for (int i = 0; i < Loss.Count; i++)
            {
                var loss = Loss[i];
                bestIndividualSum += loss;
            }
            
            print("Average population reward: " + rewardSum / Rewards.Count);
            print("Average best individual reward: " + bestIndividualSum / Loss.Count);
            
            var layoutGroup = FindObjectOfType<VerticalLayoutGroup>();
            _graphReward = Instantiate(windowGraphPrefab, layoutGroup.transform);
            _graphBestIndividualReward = Instantiate(windowGraphPrefab, layoutGroup.transform);

            StartCoroutine(ShowGraphs());
        }

        private IEnumerator ShowGraphs()
        {
            yield return new WaitForSeconds(0.1f);

            _graphReward.SetGraph(null, Rewards, GraphType.LineGraph,
                "Rewards per Episode", "episodes", "rewards");
            
            _graphBestIndividualReward.SetGraph(null, Loss, GraphType.LineGraph,
                "Best Individual Rewards per Episode", "episodes", "loss");
        }

        private void OnDestroy()
        {
            Time.timeScale = 1;
            _neModel?.Dispose();
            if (_env) _env.Close();
        }
    }
}