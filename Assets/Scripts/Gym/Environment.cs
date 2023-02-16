using System.Collections.Generic;
using Stealth_Game;
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

        [SerializeField] protected int episodeLength;
        protected int EpisodeLengthIndex;
        protected float[] CurrentObservation;
        protected List<Transform> AllEnvTransforms;
        
        private List<IResettable> _resettables;

        public int GetObservationSize => CurrentObservation.Length;

        protected virtual void Awake()
        {
            AllEnvTransforms = new List<Transform>();
            _resettables = new List<IResettable>();
            GetAllChildrenByRecursion(transform);
        }

        //TODO: Test with returning a tuple, allocating in the heap, might be faster then coping structs 
        public abstract StepInfo Step(T action);
        
        public virtual float[] ResetEnv()
        {
            EpisodeLengthIndex = 0;
            foreach (var resettable in _resettables)
            {
                resettable.ResetAgent();
            }

            return CurrentObservation;
        }

        public void Close()
        {
            // It might be better to just destroy the game-objects in the AllEnvTransforms list,
            // this way we can load new "levels" without creating a new environment class,
            // loading could be done by adding a new function in this class
            Destroy(gameObject);
        }

        private void GetAllChildrenByRecursion(Transform aParent)
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