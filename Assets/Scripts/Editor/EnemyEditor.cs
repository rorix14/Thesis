using Stealth_Game;
using UnityEditor;
using UnityEngine;

namespace Editor
{
    [CustomEditor(typeof(EnemyAgent))]
    public class EnemyEditor : UnityEditor.Editor
    {
        private Vector3 DirFromAngle(Transform agent, float angleInDegrees, bool angleIsGlobal)
        {
            if (!angleIsGlobal)
                angleInDegrees += agent.eulerAngles.y;

            return new Vector3(Mathf.Sin(angleInDegrees * Mathf.Deg2Rad), 0, Mathf.Cos(angleInDegrees * Mathf.Deg2Rad));
        }

        private void OnSceneGUI()
        {
            // draw line of sight
            foreach (var t in targets)
            {
                var enemy = (EnemyAgent)t;
                Handles.color = Color.white;
                var transform = enemy.transform;
                var position = transform.position;
                Handles.DrawWireArc(position, Vector3.up, transform.forward, 360, enemy.ViewRadius);
                Vector3 viewAngleA = DirFromAngle(transform, -enemy.ViewAngle / 2, false);
                Vector3 viewAngleB = DirFromAngle(transform, enemy.ViewAngle / 2, false);
                Handles.DrawLine(position, position + viewAngleA * enemy.ViewRadius);
                Handles.DrawLine(position, position + viewAngleB * enemy.ViewRadius);
            }
        }
    }
}