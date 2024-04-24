using System.Collections;
using System.Collections.Generic;
using Graphs;
using Gym;
using UnityEngine;
using UnityEngine.UI;

namespace TestGround
{
    public class TestHuman : MonoBehaviour
    {
        [SerializeField] private WindowGraph windowGraphPrefab;
        [SerializeField] private int numberOfEpisodes;
        [SerializeField] private float simulationSpeed;
        [SerializeField] private int skippedFrames;

        private int _episodeIndex;
        
        private int _currentSkippedFrame;
        private int _action;
        private float _reward;

        private StealthGameEnv _env;
        private Vector3 _currentPlayerAction;
        private List<float> _rewardsOverTime;
        private Dictionary<Vector3, int> _movementToAction;

        private WindowGraph _graphReward;
        private WindowGraph _graphLoss;

        private int _actionsPerformed;

        private readonly int[] _fixedActions = new[]
        {
            4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
            4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1,
            4, 1, 4, 1, 4, 1, 4, 1,  4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1, 4, 1
        };

        private void Awake()
        {
            _env = FindObjectOfType<ImageStealthGameEnv>();
            _rewardsOverTime = new List<float>(numberOfEpisodes);
            for (int i = 0; i < _rewardsOverTime.Capacity; i++)
            {
                _rewardsOverTime.Add(0f);
            }

            _movementToAction = new Dictionary<Vector3, int>
            {
                { Vector3.zero, 0 }, { Vector3.forward, 1 }, { Vector3.back, 2 }, { Vector3.right, 3 },
                { Vector3.left, 4 },
                { Vector3.forward + Vector3.right, 5 }, { Vector3.forward + Vector3.left, 6 },
                { Vector3.back + Vector3.right, 7 }, { Vector3.back + Vector3.left, 8 }, { Vector3.up, 9 },
            };
            
            skippedFrames = skippedFrames > 0 ? skippedFrames : 1;
        }

        private void Start()
        {
            var layoutGroup = FindObjectOfType<VerticalLayoutGroup>();
            _graphReward = Instantiate(windowGraphPrefab, layoutGroup.transform);
            _graphLoss = Instantiate(windowGraphPrefab, layoutGroup.transform);

            _graphReward.gameObject.SetActive(false);
            _graphLoss.gameObject.SetActive(false);

            _env.ResetEnv();
            Time.timeScale = simulationSpeed;
        }

        private void Update()
        {
            _currentPlayerAction = Input.GetKey(KeyCode.Tab)
                ? new Vector3(0, 1, 0)
                : new Vector3(Input.GetAxisRaw("Horizontal"), 0, Input.GetAxisRaw("Vertical"));
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

            var skippFrame = _currentSkippedFrame < skippedFrames - 1;

            // if (_actionsPerformed >= _fixedActions.Length) return;
            
            _action = skippFrame ? _action : _movementToAction[_currentPlayerAction];
            var stepInfo = _env.Step(_action);
            // var stepInfo = _env.Step(_movementToAction[_currentPlayerAction]);
            //var stepInfo = _env.Step(Random.Range(0, 10));
            
            _reward += stepInfo.Reward;
            ++_currentSkippedFrame;

            if (!stepInfo.Done && skippFrame) return;
            
            _rewardsOverTime[_episodeIndex] += _reward;
            _reward = 0f;
            _currentSkippedFrame = 0;

            if (!stepInfo.Done) return;

            _env.ResetEnv();
            _episodeIndex++;
        }

        private void PlotTrainingData()
        {
            Time.timeScale = 1;

            float rewardSum = 0.0f;
            foreach (var reward in _rewardsOverTime)
            {
                rewardSum += reward;
            }

            print("Average Reward: " + rewardSum / _rewardsOverTime.Count);

            _graphReward.gameObject.SetActive(true);
            _graphLoss.gameObject.SetActive(true);
            StartCoroutine(ShowGraphs());
        }

        private IEnumerator ShowGraphs()
        {
            yield return new WaitForSeconds(0.1f);

            _graphReward.SetGraph(null, _rewardsOverTime, GraphType.LineGraph,
                "Rewards per Episode", "episodes", "rewards");

            _graphLoss.SetGraph(null, _rewardsOverTime, GraphType.LineGraph,
                "Rewards per Episode", "episodes", "rewards");
        }

        private void OnDestroy()
        {
            Time.timeScale = 1;
        }
    }
}