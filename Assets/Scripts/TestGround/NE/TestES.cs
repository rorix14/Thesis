using System.Collections;
using System.Collections.Generic;
using Algorithms.NE;
using DL;
using Graphs;
using Gym;
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
        [SerializeField] private int skippedFrames;
        [SerializeField] protected float noveltyRelevance;
        
        private int _episodeIndex;
        protected JobStealthGameEnv _env;
        protected ES _neModel;

        // Cashed variables
        protected float[,] _currentSates;
        private int _currentSkippedFrame;
        private int[] _actions;

        private WindowGraph _graphReward;
        private WindowGraph _graphBestIndividualReward;

        // private Stopwatch _stopwatch;
        // private List<long> _times;

        //Only were for testing
        [SerializeField] protected ActivationFunction activationFunction;
        [SerializeField] protected float learningRate;
        [SerializeField] protected float decayRate;
        [SerializeField] protected int neuronNumber;
        [SerializeField] protected float weightsInitStd;

        public override string GetDescription()
        {
            return "ES" + (noveltyRelevance > 0 ? "-NS" : "") + ", 3 layers, " + neuronNumber + " neurons, " +
                   activationFunction + ", " + populationSize + " population size, noise std " +
                   noiseStandardDeviation + ", novelty relevance " + noveltyRelevance + ", lr " + learningRate +
                   ", decay " + decayRate + ", initialization std " + weightsInitStd;
        }

        private void Awake()
        {
            _env = FindObjectOfType<JobStealthGameEnv>();

            Rewards = new List<float>(numberOfEpisodes / requiredSimulations);
            Loss = new List<float>(numberOfEpisodes / requiredSimulations);
            for (int i = 0; i < Rewards.Capacity; i++)
            {
                Rewards.Add(0f);
                Loss.Add(0f);
            }

            if (populationSize % 2 != 0)
            {
                populationSize++;
            }
            
            skippedFrames = skippedFrames > 0 ? skippedFrames : 1;
            // _stopwatch = new Stopwatch(); 
            // _times = new List<long>(1000000);
            // Random.InitState(42);
        }

        protected virtual void Start()
        {
            _env.CreatePopulation(populationSize);
            _currentSates = _env.DistributedResetEnv();

            var network = new Layer[]
            {
                new ESNetworkLayer(AlgorithmNE.ES, populationSize, noiseStandardDeviation, _env.GetObservationSize,
                    neuronNumber, activationFunction, Instantiate(shader), paramsCoefficient: weightsInitStd),
                new ESNetworkLayer(AlgorithmNE.ES, populationSize, noiseStandardDeviation, neuronNumber, neuronNumber,
                    activationFunction, Instantiate(shader), paramsCoefficient: weightsInitStd),
                new ESNetworkLayer(AlgorithmNE.ES, populationSize, noiseStandardDeviation, neuronNumber,
                    _env.GetNumberOfActions, ActivationFunction.Linear, Instantiate(shader),
                    paramsCoefficient: weightsInitStd)
            };

            var neModel = new ESModel(network, learningRate, decayRate);
            _neModel = new ES(neModel, _env.GetNumberOfActions, populationSize, noveltyRelevance);

            Time.timeScale = simulationSpeed;
        }

        protected void FixedUpdate()
        {
            if (_episodeIndex >= numberOfEpisodes)
            {
                IsFinished = true;
                // if (!_env) return;
                // _env.Close();
                // PlotTrainingData();
                return;
            }

            //_stopwatch.Restart();

            _actions =  _currentSkippedFrame == 0 ? _neModel.SamplePopulationActions(_currentSates) : _actions;
            var stepInfo = _env.DistributedStep(_actions);

            _neModel.AddExperience(stepInfo.Rewards, stepInfo.Dones);
            _currentSates = stepInfo.Observations;
            
            _currentSkippedFrame = (_currentSkippedFrame + 1) % skippedFrames;

            if (_neModel.FinishedIndividuals < populationSize)
            {
                // _stopwatch.Stop();
                // _times.Add(_stopwatch.ElapsedMilliseconds);
                return;
            }

            if (noveltyRelevance > 0)
            {
                _neModel.DoNoveltySearch(_env.GetPlayersPositions());
            }
            
            // if (_episodeIndex % requiredSimulations == 0)
            // {
            //     _neModel.Train();
            //     Rewards[_episodeIndex / requiredSimulations] = _neModel.EpisodeRewardMean / requiredSimulations;
            //     Loss[_episodeIndex / requiredSimulations] = _neModel.EpisodeBestReward / requiredSimulations;
            // }
            // else
            // {
            //     _neModel.SoftReset();
            // }

            _neModel.Train();
            Rewards[_episodeIndex] = _neModel.EpisodeRewardMean;
            Loss[_episodeIndex] = _neModel.EpisodeBestReward;

            _currentSates = _env.DistributedResetEnv();
            _currentSkippedFrame = 0;
            ++_episodeIndex;

            // _stopwatch.Stop();
            // _times.Add(_stopwatch.ElapsedMilliseconds);
        }

        private void PlotTrainingData()
        {
            Time.timeScale = 1;

            // float timeSum = 0.0f;
            // foreach (var time in _times)
            // {
            //     timeSum += time;
            // }
            //
            // print("Average Time: " + timeSum / _times.Count);
            // EditorApplication.Beep();
            // EditorApplication.ExitPlaymode();
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
            //Time.timeScale = 1;
            _neModel?.Dispose();
            //if (_env) _env.Close();
        }
    }
}