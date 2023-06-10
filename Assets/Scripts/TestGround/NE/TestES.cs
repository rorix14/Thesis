using System.Collections;
using System.Collections.Generic;
using Algorithms.NE;
using Graphs;
using Gym;
using NN;
using UnityEngine;
using UnityEngine.UI;

namespace TestGround.NE
{
    public class TestES : MonoBehaviour
    {
        [SerializeField] protected int populationSize;
        [SerializeField] private float noiseStandardDeviation;
        [SerializeField] protected ComputeShader shader;
        [SerializeField] private WindowGraph windowGraphPrefab;
        [SerializeField] protected float simulationSpeed;
        [SerializeField] private int numberOfEpisodes;

        private int _episodeIndex;
        private JobStealthGameEnv _env;
        //private DistributedStealthGameEnv _env;

        private ES _neModel;

        private List<float> _rewardsMeanOverTime;
        private WindowGraph _graphReward;

        private float[,] _currentSates;

        private void Awake()
        {
            _env = FindObjectOfType<JobStealthGameEnv>();
            //_env = FindObjectOfType<DistributedStealthGameEnv>();
            _rewardsMeanOverTime = new List<float>(numberOfEpisodes);
            for (int i = 0; i < _rewardsMeanOverTime.Capacity; i++)
            {
                _rewardsMeanOverTime.Add(0f);
            }
        }

        protected void Start()
        {
            if (populationSize % 2 != 0)
            {
                populationSize++;
            }
            
            _env.CreatePopulation(populationSize);
            _currentSates = _env.DistributedResetEnv();

            //TODO: Make tanh function for ES layers
            var network = new NetworkLayer[]
            {
                new ESNetworkLayer(populationSize, noiseStandardDeviation, _env.GetObservationSize, 128,
                    ActivationFunction.ReLu, Instantiate(shader), true),
                new ESNetworkLayer(populationSize, noiseStandardDeviation, 128, 128, ActivationFunction.ReLu,
                    Instantiate(shader), true),
                new ESNetworkLayer(populationSize, noiseStandardDeviation, 128, _env.GetNumberOfActions,
                    ActivationFunction.Linear, Instantiate(shader), true)
            };

            var neModel = new ESModel(network, new NoLoss(Instantiate(shader)));
            _neModel = new ES(neModel, _env.GetNumberOfActions, populationSize);

            Time.timeScale = simulationSpeed;
        }

        protected void FixedUpdate()
        {
            if (_episodeIndex >= numberOfEpisodes)
            {
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

            _neModel.Train();
            // if (_episodeIndex == 260)
            // {
            //     _neModel.ReduceNoise(noiseStandardDeviation);
            // }

            _rewardsMeanOverTime[_episodeIndex] = _neModel.EpisodeRewardMean;

            _currentSates = _env.DistributedResetEnv();
            ++_episodeIndex;
        }

        private void PlotTrainingData()
        {
            Time.timeScale = 1;

            float rewardSum = 0.0f;
            foreach (var reward in _rewardsMeanOverTime)
            {
                rewardSum += reward;
            }

            // float timeSum = 0.0f;
            // foreach (var time in _times)
            // {
            //     timeSum += time;
            // }

            print("Average Reward: " + rewardSum / _rewardsMeanOverTime.Count);
            //print("Average Time: " + timeSum / _times.Count);

            var layoutGroup = FindObjectOfType<VerticalLayoutGroup>();
            _graphReward = Instantiate(windowGraphPrefab, layoutGroup.transform);
            StartCoroutine(ShowGraphs());
        }

        private IEnumerator ShowGraphs()
        {
            yield return new WaitForSeconds(0.1f);

            _graphReward.SetGraph(null, _rewardsMeanOverTime, GraphType.LineGraph,
                "Rewards per Episode", "episodes", "rewards");
        }

        private void OnDestroy()
        {
            Time.timeScale = 1;
            _neModel?.Dispose();
            if(_env) _env.Close();
        }
    }
}