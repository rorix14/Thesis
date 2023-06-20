using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Utils
{
    public static class FileHandler 
    {
        public static void SaveToJson<T>(T[] toSave, string fileName)
        {
            var content = JsonHelper.ToJson(toSave);
            WriteFile(GetPath(fileName + ".json"), content);
        }

        public static void SaveToJson<T>(T toSave, string fileName)
        {
            var content = JsonUtility.ToJson(toSave);
            WriteFile(GetPath(fileName + ".json"), content);
        }

        public static List<T> ReadListFromJson<T>(string fileName)
        {
            var content = ReadFile(GetPath(fileName+ ".json"));

            if (string.IsNullOrEmpty(content) || content == "{}")
            {
                return new List<T>();
            }

            var result = JsonHelper.FromJson<T>(content).ToList();
            return result;
        }

        public static T ReadFromJson<T>(string fileName)
        {
            var content = ReadFile(GetPath(fileName+ ".json"));

            if (string.IsNullOrEmpty(content) || content == "{}")
            {
                return default;
            }

            var result = JsonUtility.FromJson<T>(content);
            return result;
        }

        private static string GetPath(string fileName)
        {
            return Application.dataPath + "/" + fileName;
        }

        private static void WriteFile(string path, string content)
        {
            var fileStream = new FileStream(path, FileMode.Create);
            using var writer = new StreamWriter(fileStream);
            writer.Write(content);
        }

        private static string ReadFile(string path)
        {
            if (File.Exists(path)) return "";

            using var reader = new StreamReader(path);
            var content = reader.ReadToEnd();
            return content;
        }
    }

    public static class JsonHelper
    {
        public static T[] FromJson<T>(string json)
        {
            var wrapper = JsonUtility.FromJson<Wrapper<T>>(json);
            return wrapper.items;
        }

        public static string ToJson<T>(T[] array, bool prettyPrint = true)
        {
            var wrapper = new Wrapper<T>
            {
                items = array
            };

            return JsonUtility.ToJson(wrapper, prettyPrint);
        }
        
        [Serializable]
        private class Wrapper<T>
        {
            public T[] items;
        }
    }
}
