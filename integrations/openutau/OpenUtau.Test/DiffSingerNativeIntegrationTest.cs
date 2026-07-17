using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NAudio.Wave;
using Newtonsoft.Json;
using OpenUtau.Api;
using OpenUtau.Classic;
using OpenUtau.Core;
using OpenUtau.Core.DiffSinger;
using OpenUtau.Core.Render;
using OpenUtau.Core.SignalChain;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Util;
using Xunit;

namespace OpenUtau.Test {
    public class DiffSingerNativeIntegrationTest {
        [Fact]
        public void OfficialDiffSingerRendererLoadsUstAndExportsWave() {
            var ust = Required("GYU_DIFFSINGER_UST");
            var singerId = Required("GYU_DIFFSINGER_SINGER_ID");
            var output = Required("GYU_DIFFSINGER_OUTPUT");
            var metrics = Required("GYU_DIFFSINGER_METRICS");
            var maxNotes = int.Parse(Environment.GetEnvironmentVariable("GYU_DIFFSINGER_MAX_NOTES") ?? "0");
            var steps = int.Parse(Environment.GetEnvironmentVariable("GYU_DIFFSINGER_STEPS") ?? "20");
            var depth = double.Parse(Environment.GetEnvironmentVariable("GYU_DIFFSINGER_DEPTH") ?? "0.6");
            var tempoScale = double.Parse(Environment.GetEnvironmentVariable("GYU_DIFFSINGER_TEMPO_SCALE") ?? "1.0");

            var scheduler = new PumpTaskScheduler();
            Encoding.RegisterProvider(CodePagesEncodingProvider.Instance);
            DocManager.Inst.Initialize(Thread.CurrentThread, scheduler);
            DocManager.Inst.PostOnUIThread = scheduler.Post;
            SingerManager.Inst.Initialize();
            var project = Ust.Load(new[] { ust });
            foreach (var tempo in project.tempos) {
                tempo.bpm /= tempoScale;
            }
            var part = Assert.Single(project.parts.OfType<UVoicePart>());
            if (maxNotes > 0 && part.notes.Count > maxNotes) {
                foreach (var note in part.notes.Skip(maxNotes).ToArray()) {
                    part.notes.Remove(note);
                }
                part.Duration = part.notes.Last().End + project.resolution;
            }
            if (Environment.GetEnvironmentVariable("GYU_DIFFSINGER_KEEP_LEADING") != "1") {
                var notes = part.notes.ToArray();
                var shift = notes.First().position - project.resolution;
                part.notes.Clear();
                foreach (var note in notes) {
                    note.position -= shift;
                    part.notes.Add(note);
                }
                part.Duration = part.notes.Last().End + project.resolution;
            }
            var track = Assert.Single(project.tracks);
            var singer = SingerManager.Inst.GetSinger(singerId);
            Assert.NotNull(singer);
            Assert.Equal(USingerType.DiffSinger, singer.SingerType);
            var diffSinger = Assert.IsType<DiffSingerSinger>(singer);
            track.Singer = singer;
            track.singer = singer.Id;
            track.Phonemizer = PhonemizerFactory.Get(typeof(DiffSingerJapanesePhonemizer)).Create();
            track.phonemizer = typeof(DiffSingerJapanesePhonemizer).FullName;
            track.RendererSettings.renderer = Renderers.DIFFSINGER;
            Preferences.Default.DiffSingerSteps = steps;
            Preferences.Default.DiffSingerDepth = depth;
            var effectiveDepth = diffSinger.dsConfig.useVariableDepth
                ? Math.Min(depth, diffSinger.dsConfig.maxDepth)
                : 1.0;

            Directory.CreateDirectory(PathManager.Inst.CachePath);
            DocManager.Inst.ExecuteCmd(new LoadProjectNotification(project));
            project.ValidateFull();
            WaitForPhonemes(project, part, scheduler);
            var internalOnsets = Environment.GetEnvironmentVariable("GYU_DIFFSINGER_INTERNAL_ONSETS") == "1";
            if (internalOnsets) {
                ApplyInternalOnsets(project, part);
                project.Validate(new ValidateOptions { Part = part, SkipTiming = true, SkipPhonemizer = true });
            }
            Assert.NotEmpty(part.phonemes);
            var errors = part.phonemes.Where(phoneme => phoneme.Error).ToArray();
            Assert.True(errors.Length == 0, string.Join(" | ", errors.Take(10).Select(
                phoneme => $"{phoneme}: {phoneme.ErrorException}")));
            Directory.CreateDirectory(Path.GetDirectoryName(output));
            Directory.CreateDirectory(Path.GetDirectoryName(metrics));
            var timer = Stopwatch.StartNew();
            var mix = RenderAndPump(project, scheduler);
            WaveFileWriter.CreateWaveFile16(output, new ExportAdapter(mix));
            timer.Stop();
            using var reader = new WaveFileReader(output);
            var report = new Dictionary<string, object> {
                ["official_openutau_diffsinger_renderer"] = true,
                ["singer_id"] = singer.Id,
                ["singer_name"] = singer.Name,
                ["singer_type"] = singer.SingerType.ToString(),
                ["notes"] = part.notes.Count,
                ["phonemes"] = part.phonemes.Count,
                ["phoneme_errors"] = part.phonemes.Count(phone => phone.Error),
                ["steps"] = steps,
                ["requested_depth"] = depth,
                ["depth"] = effectiveDepth,
                ["tempo_scale"] = tempoScale,
                ["internal_onsets"] = internalOnsets,
                ["render_seconds"] = timer.Elapsed.TotalSeconds,
                ["sample_rate"] = reader.WaveFormat.SampleRate,
                ["channels"] = reader.WaveFormat.Channels,
                ["duration_seconds"] = reader.TotalTime.TotalSeconds,
                ["output"] = output,
                ["score_notes"] = part.notes.Select(note => new {
                    lyric = note.lyric,
                    tone = note.tone,
                    start_tick = note.position,
                    end_tick = note.End,
                    start_ms = project.timeAxis.TickPosToMsPos(note.position),
                    end_ms = project.timeAxis.TickPosToMsPos(note.End),
                }).ToArray(),
                ["phoneme_timeline"] = part.phonemes.Select(phoneme => new {
                    phoneme = phoneme.phoneme,
                    lyric = phoneme.Parent.lyric,
                    tone = phoneme.Parent.tone,
                    start_tick = phoneme.position,
                    end_tick = phoneme.End,
                    start_ms = phoneme.PositionMs,
                    end_ms = phoneme.EndMs,
                }).ToArray(),
            };
            File.WriteAllText(metrics, JsonConvert.SerializeObject(report, Formatting.Indented));
        }

        static string Required(string name) {
            var value = Environment.GetEnvironmentVariable(name);
            Assert.False(string.IsNullOrWhiteSpace(value), $"set {name}");
            return value;
        }

        static void WaitForPhonemes(UProject project, UVoicePart part, PumpTaskScheduler scheduler) {
            var deadline = DateTime.UtcNow.AddMinutes(3);
            while ((part.phonemes.Count == 0 || part.phonemes.Any(phone => phone.phoneme == null))
                    && DateTime.UtcNow < deadline) {
                scheduler.RunPending();
                project.ValidateFull();
                Thread.Sleep(20);
            }
            scheduler.RunPending();
        }

        static void ApplyInternalOnsets(UProject project, UVoicePart part) {
            foreach (var note in part.notes) {
                var phones = part.phonemes.Where(phone => phone.Parent == note).OrderBy(phone => phone.index).ToArray();
                if (phones.Length == 0) {
                    continue;
                }
                var noteStartMs = project.timeAxis.TickPosToMsPos(note.position);
                var noteEndMs = project.timeAxis.TickPosToMsPos(note.End);
                var onsetMs = Math.Min(40, (noteEndMs - noteStartMs) * .4);
                for (var index = 0; index < phones.Length; index++) {
                    var fraction = phones.Length == 1 ? 0 : (double)index / (phones.Length - 1);
                    var desired = project.timeAxis.MsPosToTickPos(noteStartMs + onsetMs * fraction);
                    note.GetPhonemeOverride(phones[index].index).offset = desired - phones[index].rawPosition;
                }
            }
        }

        static ISignalSource RenderAndPump(UProject project, PumpTaskScheduler scheduler) {
            CancellationTokenSource cancellation = null;
            var task = Task.Run(() => new RenderEngine(project)
                .RenderMixdown(TaskScheduler.Default, ref cancellation, wait: true).Item1);
            while (!task.IsCompleted) {
                scheduler.RunPending();
                Thread.Sleep(20);
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
