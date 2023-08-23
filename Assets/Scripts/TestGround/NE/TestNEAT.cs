using System.Collections;
using System.Collections.Generic;
using Algorithms.NE.NEAT;
using Graphs;
using Gym;
using TestGround.Base;
using UnityEngine;
using UnityEngine.UI;

namespace TestGround.NE
{
    public class TestNEAT : TestAlgorithmBase
    {
        [SerializeField] private WindowGraph windowGraphPrefab;
        [SerializeField] protected float simulationSpeed;
        [SerializeField] private int numberOfEpisodes;

        [SerializeField] private int populationSize;
        [SerializeField] private float mutationNoiseStd;

        private int _episodeIndex;
        private JobStealthGameEnv _env;
        private NEAT _neat;

        private List<float> _rewardsMeanOverTime;
        private List<float> _bestIndividualRewardsMeanOverTime;
        private WindowGraph _graphReward;
        private WindowGraph _graphBestIndividualReward;

        private float[,] _currentSates;

        public override string GetDescription()
        {
            return "";
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
        }

        void Start()
        {
            Random.InitState(42);

            _env.CreatePopulation(populationSize);
            _currentSates = _env.DistributedResetEnv();

            var neatModel = new NEATModel(populationSize, _env.GetObservationSize, _env.GetNumberOfActions);
            _neat = new NEAT(neatModel, _env.GetNumberOfActions, populationSize);
            Time.timeScale = simulationSpeed;
        }

        private void FixedUpdate()
        {
            if (_episodeIndex >= numberOfEpisodes)
            {
                if (!_env) return;

                _env.Close();
                PlotTrainingData();
                return;
            }

            var actions = _neat.SamplePopulationActions(_currentSates);
            var stepInfo = _env.DistributedStep(actions);

            _neat.AddExperience(stepInfo.Rewards, stepInfo.Dones);
            _currentSates = stepInfo.Observations;

            if (_neat.FinishedIndividuals < populationSize) return;

            _neat.Train();

            _rewardsMeanOverTime[_episodeIndex] = _neat.EpisodeRewardMean;

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

            float bestIndividualSum = 0.0f;
            for (int i = 0; i < Loss.Count; i++)
            {
                var loss = Loss[i];
                bestIndividualSum += loss;
            }

            print("Average population reward: " + rewardSum / _rewardsMeanOverTime.Count);
            print("Average best individual reward: " + bestIndividualSum / Loss.Count);

            var layoutGroup = FindObjectOfType<VerticalLayoutGroup>();
            _graphReward = Instantiate(windowGraphPrefab, layoutGroup.transform);
            _graphBestIndividualReward = Instantiate(windowGraphPrefab, layoutGroup.transform);

            StartCoroutine(ShowGraphs());
        }

        private IEnumerator ShowGraphs()
        {
            yield return new WaitForSeconds(0.1f);

            _graphReward.SetGraph(null, _rewardsMeanOverTime, GraphType.LineGraph,
                "Population Rewards per Episode", "episodes", "rewards");

            _graphBestIndividualReward.SetGraph(null, Loss, GraphType.LineGraph,
                "Best Individual Rewards per Episode", "episodes", "loss");
        }

        private void OnDestroy()
        {
            Time.timeScale = 1;
            _neat?.Dispose();
            if (_env) _env.Close();
        }
    }
}