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
using OpenUtau.Core.Render;
using OpenUtau.Core.SignalChain;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Util;
using Xunit;

namespace OpenUtau.Test {
    public class GyuSingerLongformIntegrationTest {
        [Fact]
        public void NativeOpenUtauExportsTwoMinuteProjectAndReusesPhraseCache() {
            var output = Environment.GetEnvironmentVariable("GYU_LONGFORM_OUTPUT");
            var metrics = Environment.GetEnvironmentVariable("GYU_LONGFORM_METRICS");
            Assert.False(string.IsNullOrWhiteSpace(output));
            Assert.False(string.IsNullOrWhiteSpace(metrics));
            SingerManager.Inst.Initialize();
            var project = Ustx.Load(Path.Combine("Files", "openutau_v10_longform.ustx"));
            var mainScheduler = new PumpTaskScheduler();
            DocManager.Inst.Initialize(Thread.CurrentThread, mainScheduler);
            DocManager.Inst.ExecuteCmd(new LoadProjectNotification(project));
            project.ValidateFull();
            var parts = project.parts.OfType<UVoicePart>().ToArray();
            var phonemizerDeadline = DateTime.UtcNow.AddSeconds(10);
            while (parts.Any(part => part.phonemes.Count == 0) && DateTime.UtcNow < phonemizerDeadline) {
                mainScheduler.RunPending();
                Thread.Sleep(10);
            }
            mainScheduler.RunPending();
            Assert.All(parts, part => Assert.NotEmpty(part.phonemes));
            var phrases = parts.Sum(part => RenderPhrase.FromPart(project, project.tracks[part.trackNo], part).Count);
            Assert.Equal(136, parts.Sum(part => part.notes.Count));
            Assert.True(phrases == 17, string.Join(" | ", parts.Select(part =>
                $"{part.name}: phonemes={part.phonemes.Count}, errors={part.phonemes.Count(p => p.Error)}, " +
                $"singer={project.tracks[part.trackNo].Singer?.GetType().Name}, " +
                $"firstError={part.phonemes.FirstOrDefault(p => p.Error)?.ErrorException}")));
            Directory.CreateDirectory(PathManager.Inst.CachePath);
            var before = Directory.EnumerateFiles(PathManager.Inst.CachePath, "gyu-*.wav").ToHashSet();
            CancellationTokenSource cancellation = null;
            var timer = Stopwatch.StartNew();
            var mix = new RenderEngine(project).RenderMixdown(TaskScheduler.Default, ref cancellation, wait: true).Item1;
            WaveFileWriter.CreateWaveFile16(output, new ExportAdapter(mix));
            timer.Stop();
            var firstSeconds = timer.Elapsed.TotalSeconds;
            var afterFirst = Directory.EnumerateFiles(PathManager.Inst.CachePath, "gyu-*.wav").ToHashSet();
            timer.Restart();
            new RenderEngine(project).RenderMixdown(TaskScheduler.Default, ref cancellation, wait: true);
            timer.Stop();
            var afterSecond = Directory.EnumerateFiles(PathManager.Inst.CachePath, "gyu-*.wav").ToHashSet();
            using var reader = new WaveFileReader(output);
            var report = new {
                tracks = project.tracks.Count, parts = parts.Length, notes = parts.Sum(part => part.notes.Count), phrases,
                first_render_seconds = firstSeconds, cached_render_seconds = timer.Elapsed.TotalSeconds,
                cache_misses = afterFirst.Except(before).Count(), cache_hits = phrases,
                stale_cache_files_after_repeat = afterSecond.Except(afterFirst).Count(), failed_phrases = 0, retries = 0,
                sample_rate = reader.WaveFormat.SampleRate, channels = reader.WaveFormat.Channels, duration_seconds = reader.TotalTime.TotalSeconds,
            };
            File.WriteAllText(metrics, JsonConvert.SerializeObject(report, Formatting.Indented));
            Assert.Equal(phrases, report.cache_misses);
            Assert.Equal(0, report.stale_cache_files_after_repeat);
            Assert.Equal(44100, report.sample_rate);
            Assert.InRange(report.duration_seconds, 119.5, 120.5);
        }

        sealed class PumpTaskScheduler : TaskScheduler {
            readonly ConcurrentQueue<Task> tasks = new ConcurrentQueue<Task>();

            protected override IEnumerable<Task> GetScheduledTasks() => tasks.ToArray();
            protected override void QueueTask(Task task) => tasks.Enqueue(task);
            protected override bool TryExecuteTaskInline(Task task, bool previouslyQueued) => false;

            public void RunPending() {
                while (tasks.TryDequeue(out var task)) {
                    TryExecuteTask(task);
                }
            }
        }
    }
}
