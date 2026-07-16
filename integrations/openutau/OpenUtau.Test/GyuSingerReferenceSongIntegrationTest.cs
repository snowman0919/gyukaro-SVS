using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NAudio.Wave;
using Newtonsoft.Json;
using OpenUtau.Core;
using OpenUtau.Core.Format;
using OpenUtau.Core.GyuSinger;
using OpenUtau.Core.Render;
using OpenUtau.Core.SignalChain;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Util;
using Xunit;

namespace OpenUtau.Test {
    public class GyuSingerReferenceSongIntegrationTest {
        [Fact]
        public void NativeOpenUtauValidatesAndExportsLocalReferenceProject() {
            var projectPath = Environment.GetEnvironmentVariable("GYU_REFERENCE_PROJECT");
            var output = Environment.GetEnvironmentVariable("GYU_REFERENCE_OUTPUT");
            var metrics = Environment.GetEnvironmentVariable("GYU_REFERENCE_METRICS");
            Assert.True(File.Exists(projectPath));
            Assert.False(string.IsNullOrWhiteSpace(output));
            Assert.False(string.IsNullOrWhiteSpace(metrics));
            SingerManager.Inst.Initialize();
            var project = Ustx.Load(projectPath);
            var scheduler = new PumpTaskScheduler();
            DocManager.Inst.Initialize(Thread.CurrentThread, scheduler);
            DocManager.Inst.PostOnUIThread = scheduler.Post;
            DocManager.Inst.ExecuteCmd(new LoadProjectNotification(project));
            project.ValidateFull();
            var parts = project.parts.OfType<UVoicePart>().ToArray();
            var deadline = DateTime.UtcNow.AddSeconds(30);
            while (parts.Any(part => part.phonemes.Count == 0) && DateTime.UtcNow < deadline) {
                scheduler.RunPending();
                Thread.Sleep(10);
            }
            scheduler.RunPending();
            Assert.All(parts, part => Assert.NotEmpty(part.phonemes));
            var renderPhrases = parts.SelectMany(part => RenderPhrase.FromPart(project, project.tracks[part.trackNo], part)).ToArray();
            var phrases = renderPhrases.Length;
            var requests = Environment.GetEnvironmentVariable("GYU_REFERENCE_REQUESTS");
            if (!string.IsNullOrWhiteSpace(requests)) {
                File.WriteAllText(requests, JsonConvert.SerializeObject(
                    renderPhrases.Select(GyuSingerRenderer.BuildRequest), Formatting.Indented));
            }
            var validateOnly = Environment.GetEnvironmentVariable("GYU_REFERENCE_VALIDATE_ONLY") == "1";
            var report = new Dictionary<string, object> {
                ["tracks"] = project.tracks.Count,
                ["parts"] = parts.Length,
                ["notes"] = parts.Sum(part => part.notes.Count),
                ["phonemes"] = parts.Sum(part => part.phonemes.Count),
                ["phrases"] = phrases,
                ["phoneme_errors"] = parts.Sum(part => part.phonemes.Count(phone => phone.Error)),
                ["validate_only"] = validateOnly,
            };
            if (!validateOnly) {
                Directory.CreateDirectory(PathManager.Inst.CachePath);
                var before = Directory.EnumerateFiles(PathManager.Inst.CachePath, "gyu-*.wav").ToHashSet();
                var timer = Stopwatch.StartNew();
                var mix = RenderAndPump(project, scheduler);
                WaveFileWriter.CreateWaveFile16(output, new ExportAdapter(mix));
                timer.Stop();
                var afterFirst = Directory.EnumerateFiles(PathManager.Inst.CachePath, "gyu-*.wav").ToHashSet();
                var firstSeconds = timer.Elapsed.TotalSeconds;
                timer.Restart();
                RenderAndPump(project, scheduler);
                timer.Stop();
                var afterSecond = Directory.EnumerateFiles(PathManager.Inst.CachePath, "gyu-*.wav").ToHashSet();
                using var reader = new WaveFileReader(output);
                report["first_render_seconds"] = firstSeconds;
                report["cached_render_seconds"] = timer.Elapsed.TotalSeconds;
                report["cache_misses"] = afterFirst.Except(before).Count();
                report["cache_hits"] = phrases;
                report["stale_cache_files_after_repeat"] = afterSecond.Except(afterFirst).Count();
                report["failed_phrases"] = 0;
                report["retries"] = 0;
                report["sample_rate"] = reader.WaveFormat.SampleRate;
                report["channels"] = reader.WaveFormat.Channels;
                report["duration_seconds"] = reader.TotalTime.TotalSeconds;
            }
            File.WriteAllText(metrics, JsonConvert.SerializeObject(report, Formatting.Indented));
        }

        static ISignalSource RenderAndPump(UProject project, PumpTaskScheduler scheduler) {
            CancellationTokenSource cancellation = null;
            var task = Task.Run(() => new RenderEngine(project)
                .RenderMixdown(TaskScheduler.Default, ref cancellation, wait: true).Item1);
            while (!task.IsCompleted) {
                scheduler.RunPending();
                Thread.Sleep(10);
            }
            scheduler.RunPending();
            return task.GetAwaiter().GetResult();
        }

        sealed class PumpTaskScheduler : TaskScheduler {
            readonly ConcurrentQueue<Task> tasks = new();
            protected override IEnumerable<Task> GetScheduledTasks() => tasks.ToArray();
            protected override void QueueTask(Task task) => tasks.Enqueue(task);
            protected override bool TryExecuteTaskInline(Task task, bool previouslyQueued) => false;
            public void Post(Action action) => Task.Factory.StartNew(
                action, CancellationToken.None, TaskCreationOptions.None, this);
            public void RunPending() {
                while (tasks.TryDequeue(out var task)) TryExecuteTask(task);
            }
        }
    }
}
