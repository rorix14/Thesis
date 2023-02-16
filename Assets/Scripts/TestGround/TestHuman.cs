using System.Collections.Generic;
using Graphs;
using Gym;
using UnityEngine;

namespace TestGround
{
    public class TestHuman : MonoBehaviour
    {
        [SerializeField] private WindowGraph windowGraphPrefab;
        [SerializeField] private int numberOfEpisodes;
        private int _episodeIndex;

        private StealthGameEnv _env;
        private Vector3 _currentPlayerAction;
        private List<float> _rewardsOverTime;

        private void Awake()
        {
            _env = FindObjectOfType<StealthGameEnv>();
            _rewardsOverTime = new List<float>(numberOfEpisodes);
            for (int i = 0; i < _rewardsOverTime.Capacity; i++)
            {
                _rewardsOverTime.Add(0f);
            }
        }

        private void Start()
        {
            _env.ResetEnv();
        }

        private void Update()
        {
            _currentPlayerAction = new Vector3(Input.GetAxisRaw("Horizontal"), 0, Input.GetAxisRaw("Vertical"));
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

            var stepInfo = _env.Step(_currentPlayerAction);
            _rewardsOverTime[_episodeIndex] += stepInfo.Reward;

            if (!stepInfo.Done) return;

            _env.ResetEnv();
            _episodeIndex++;
        }

        private void PlotTrainingData()
        {
            var canvas = FindObjectOfType<Canvas>();
            var graph = Instantiate(windowGraphPrefab, canvas.transform);
            graph.SetGraph(null, _rewardsOverTime, GraphType.LineGraph, 
                "Rewards per Episode", "time", "reward");
        }
    }
}