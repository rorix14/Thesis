using System.Collections.Generic;
using Stealth_Game;
using UnityEditor;
using UnityEngine;

namespace Gym
{
    public abstract class Environment<T> : MonoBehaviour
    {
        public struct StepInfo
        {
            public readonly float[] Observation;
            public float Reward;
            public bool Done;

            public StepInfo(float[] observation, float reward, bool done)
            {
                Observation = observation;
                Reward = reward;
                Done = done;
            }
        }

        public struct DistributedStepInfo
        {
            public readonly float[,] Observations;
            public float[] Rewards;
            public bool[] Dones;

            public DistributedStepInfo(float[,] observations, float[] rewards, bool[] dones)
            {
                Observations = observations;
                Rewards = rewards;
                Dones = dones;
            }
        }

        [SerializeField] private Vector4 levelSizeMaxMin;
        [SerializeField] protected int episodeLength;
        protected int EpisodeLengthIndex;
        protected int ObservationLenght;
        protected T[] ActionLookup;
        protected List<Transform> AllEnvTransforms;

        private Vector2 _levelSizeRange;
        private float _levelMaxDistance;
        protected List<IResettable> _resettables;

        public int GetObservationSize => ObservationLenght;
        public int GetNumberOfActions => ActionLookup.Length;

        protected virtual void Awake()
        {
            AllEnvTransforms = new List<Transform>();
            _resettables = new List<IResettable>();
            GetAllChildrenByRecursion(transform);

            _levelSizeRange = new Vector2(levelSizeMaxMin.x - levelSizeMaxMin.y, levelSizeMaxMin.z - levelSizeMaxMin.w);
            _levelMaxDistance =
                Mathf.Sqrt(_levelSizeRange.x * _levelSizeRange.x + _levelSizeRange.y * _levelSizeRange.y);
            
            // float[] myArray = { 2, 3, 4, 5, 6, 7, 8, 9, 10 };
            // for (int i = 0; i < myArray.Length; i++)
            // {
            //     print(2.0f * (myArray[i] - (2)) / 8 - 1.0f);
            // }
        }

        //TODO: Test with returning a tuple, allocating in the heap, might be faster then copying structs 
        public abstract StepInfo Step(int action);

        public virtual DistributedStepInfo DistributedStep(int[] actions)
        {
            return new DistributedStepInfo();
        }


        public virtual float[] ResetEnv()
        {
            EpisodeLengthIndex = 0;
            foreach (var resettable in _resettables)
            {
                resettable.ResetAgent();
            }

            return null;
        }

        public virtual float[,] DistributedResetEnv()
        {
            EpisodeLengthIndex = 0;
            foreach (var resettable in _resettables)
            {
                resettable.ResetAgent();
            }

            return null;
        }

        public virtual void Close()
        {
            // It might be better to just destroy the game-objects in the AllEnvTransforms list,
            // this way we can load new "levels" without creating a new environment class,
            // loading could be done by adding a new function in this class
            Destroy(gameObject);
        }

        // Normalize values between 0 and 1
        protected float NormalizeDistance(float value)
        {
            return value / _levelMaxDistance;
        }

        // Normalize values between -1 and 1
        protected float NormalizePosition(float value, bool isX)
        {
            float range = _levelSizeRange.y;
            float min = levelSizeMaxMin.w;
            if (isX)
            {
                range = _levelSizeRange.x;
                min = levelSizeMaxMin.y;
            }
            
            return 2.0f * (value - min) / range - 1.0f;
        }

        protected void GetAllChildrenByRecursion(Transform aParent)
        {
            foreach (Transform child in aParent)
            {
                AllEnvTransforms.Add(child);
                var resettable = child.GetComponent<IResettable>();
                if (!(resettable is null))
                {
                    _resettables.Add(resettable);
                }

                GetAllChildrenByRecursion(child);
            }
        }
    }
}