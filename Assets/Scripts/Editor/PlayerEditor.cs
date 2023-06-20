using Stealth_Game;
using UnityEditor;
using UnityEngine;

namespace Editor
{
    [CustomEditor(typeof(PlayerAgent))]
    public class PlayerEditor : UnityEditor.Editor
    {
        private void OnSceneGUI()
        {
            var player = (PlayerAgent)target;
            Handles.color = Color.red;

            if (player.ViewPoints is null) return;

            var playerPos = player.transform.position;
            foreach (var viewPoint in player.ViewPoints)
            {
                Handles.DrawLine(playerPos, viewPoint);
            }
        }
    }
}