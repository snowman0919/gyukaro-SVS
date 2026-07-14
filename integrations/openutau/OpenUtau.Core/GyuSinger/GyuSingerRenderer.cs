using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using NAudio.Wave;
using Newtonsoft.Json;
using OpenUtau.Core.Format;
using OpenUtau.Core.Render;
using OpenUtau.Core.SignalChain;
using OpenUtau.Core.Ustx;
using OpenUtau.Core.Util;
using Serilog;

namespace OpenUtau.Core.GyuSinger {
    public sealed class GyuSinger : USinger {
        static readonly List<USubbank> EmptySubbanks = new();
        static readonly List<string> EmptyErrors = new();
        public GyuSinger() { found = loaded = true; }
        public override string Id => "GYU-SINGER";
        public override string Name => "GYU Singer";
        public override USingerType SingerType => USingerType.GyuSinger;
        public override string Author => "gyukaro";
        public override string Voice => "GYU";
        public override string Version => "0.9";
        public override string OtherInfo => "Resident phrase neural renderer";
        public override string DefaultPhonemizer => "OpenUtau.Core.DefaultPhonemizer";
        public override IList<USubbank> Subbanks => EmptySubbanks;
        public override IList<string> Errors => EmptyErrors;
        public override bool TryGetOto(string phoneme, out UOto oto) { oto = UOto.OfDummy(phoneme); return true; }
    }

    public static class GyuSingerLoader {
        public static IEnumerable<USinger> FindAllSingers() => new[] { new GyuSinger() };
    }

    public sealed class GyuSingerRenderer : IRenderer {
        const int CurveTickInterval = 5;
        const string StyleCurve = "gyus";
        static readonly string[] StylePresets = { "neutral", "soft", "breathy", "energetic", "dark", "bright" };
        static readonly HashSet<string> SupportedExpressions = new() {
            OpenUtau.Core.Format.Ustx.DYN,
            OpenUtau.Core.Format.Ustx.PITD,
            OpenUtau.Core.Format.Ustx.BREC,
            OpenUtau.Core.Format.Ustx.TENC,
            StyleCurve,
        };
        static readonly HttpClient Client = new() { Timeout = TimeSpan.FromMinutes(10) };
        static readonly object RenderLock = new();

        public USingerType SingerType => USingerType.GyuSinger;
        public bool SupportsRenderPitch => true;
        public bool SupportsRealCurve => false;

        public bool SupportsExpression(UExpressionDescriptor descriptor) => SupportedExpressions.Contains(descriptor.abbr);

        public RenderResult Layout(RenderPhrase phrase) => new() {
            leadingMs = 0,
            positionMs = phrase.positionMs,
            estimatedLengthMs = phrase.durationMs,
        };

        public Task<RenderResult> Render(RenderPhrase phrase, Progress progress, int trackNo, CancellationTokenSource cancellation, bool isPreRender = false) {
            return Task.Run(() => {
                lock (RenderLock) {
                    cancellation.Token.ThrowIfCancellationRequested();
                    var result = Layout(phrase);
                    var request = BuildRequest(phrase);
                    var json = JsonConvert.SerializeObject(request);
                    var digest = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(json))).ToLowerInvariant()[..24];
                    Directory.CreateDirectory(PathManager.Inst.CachePath);
                    var wavPath = Path.Join(PathManager.Inst.CachePath, $"gyu-{digest}.wav");
                    phrase.AddCacheFile(wavPath);
                    var progressInfo = $"Track {trackNo + 1}: GYU-SINGER \"{string.Join(" ", phrase.notes.Select(note => note.lyric))}\"";
                    progress.Complete(0, progressInfo);
                    try {
                        if (!File.Exists(wavPath)) {
                            var url = Environment.GetEnvironmentVariable("GYU_RENDERER_URL") ?? "http://127.0.0.1:8765/render";
                            using var content = new StringContent(json, Encoding.UTF8, "application/json");
                            using var response = Client.PostAsync(url, content, cancellation.Token).GetAwaiter().GetResult();
                            response.EnsureSuccessStatusCode();
                            var bytes = response.Content.ReadAsByteArrayAsync(cancellation.Token).GetAwaiter().GetResult();
                            var temporary = wavPath + ".tmp";
                            File.WriteAllBytes(temporary, bytes);
                            File.Move(temporary, wavPath, true);
                        }
                        result.samples = LoadEditorSamples(wavPath);
                        progress.Complete(phrase.phones.Length, progressInfo);
                        return result;
                    } catch (Exception error) {
                        Log.Error(error, "GYU-SINGER phrase render failed.");
                        throw new InvalidOperationException("GYU-SINGER resident renderer failed. Check GYU_RENDERER_URL and /health.", error);
                    }
                }
            }, cancellation.Token);
        }

        static float[] LoadEditorSamples(string wavPath) {
            using var waveStream = Wave.OpenFile(wavPath);
            // Wave.GetSamples performs the single OpenUtau-standard conversion to 44.1 kHz.
            return Wave.GetSamples(waveStream.ToSampleProvider().ToMono(1, 0));
        }

        internal static GyuRequest BuildRequest(RenderPhrase phrase) {
            var notes = phrase.notes
                .Where(note => note.positionMs >= phrase.positionMs && note.positionMs < phrase.endMs)
                .Select(note => new GyuNote {
                    pitch = note.adjustedTone,
                    start = Math.Max(0, (note.positionMs - phrase.positionMs) / 1000.0),
                    duration = note.durationMs / 1000.0,
                    lyric = note.lyric,
                }).ToList();
            if (notes.Count == 0) {
                throw new InvalidOperationException("GYU-SINGER received an empty RenderPhrase.");
            }
            var pitch = new List<GyuPoint>();
            var dynamics = new List<GyuPoint>();
            var breathiness = new List<GyuPoint>();
            var tension = new List<GyuPoint>();
            for (var index = 0; index < phrase.pitches.Length; index++) {
                var tick = phrase.position - phrase.leading + index * CurveTickInterval;
                var time = (phrase.timeAxis.TickPosToMsPos(tick) - phrase.positionMs) / 1000.0;
                if (time < 0 || time > phrase.durationMs / 1000.0) continue;
                var note = notes.LastOrDefault(value => value.start <= time && time < value.start + value.duration) ?? notes[0];
                pitch.Add(new GyuPoint { time = time, value = phrase.pitches[index] * 0.01 - note.pitch });
                if (phrase.dynamics != null) dynamics.Add(new GyuPoint { time = time, value = phrase.dynamics[index] });
                if (phrase.breathiness != null) breathiness.Add(new GyuPoint { time = time, value = phrase.breathiness[index] * 0.01 });
                if (phrase.tension != null) tension.Add(new GyuPoint { time = time, value = phrase.tension[index] * 0.01 });
            }
            var customStyle = phrase.curves.FirstOrDefault(curve => curve.Item1 == StyleCurve)?.Item2;
            var styleIndex = customStyle == null ? 0 : Math.Clamp((int)Math.Round(customStyle.Average()), 0, StylePresets.Length - 1);
            return new GyuRequest {
                protocol = "gyu-renderer-v2",
                source = "OpenUtau RenderPhrase",
                language = DetectLanguage(string.Concat(notes.Select(note => note.lyric))),
                tempo = phrase.phones.FirstOrDefault()?.tempo ?? 120,
                sample_rate = 48000,
                notes = notes,
                phonemes = phrase.phones.Select(phone => new GyuPhoneme {
                    phoneme = phone.phoneme,
                    start = Math.Max(0, (phone.positionMs - phrase.positionMs) / 1000.0),
                    duration = phone.durationMs / 1000.0,
                }).ToList(),
                curves = new Dictionary<string, List<GyuPoint>> {
                    { "pitch", pitch }, { "dynamics", dynamics }, { "breathiness", breathiness }, { "tension", tension },
                },
                style = new Dictionary<string, object> { { "preset", StylePresets[styleIndex] } },
            };
        }

        internal static string DetectLanguage(string lyrics) {
            if (lyrics.Any(value => value >= '\uac00' && value <= '\ud7af')) return "ko";
            if (lyrics.Any(value => (value >= '\u3040' && value <= '\u30ff') || (value >= '\u4e00' && value <= '\u9fff'))) return "ja";
            return "en";
        }

        public RenderPitchResult LoadRenderedPitch(RenderPhrase phrase) {
            return new RenderPitchResult {
                ticks = Enumerable.Range(0, phrase.pitches.Length).Select(index => (float)(index * CurveTickInterval - phrase.leading)).ToArray(),
                tones = phrase.pitches.Select(value => value * 0.01f).ToArray(),
            };
        }

        public List<RenderRealCurveResult> LoadRenderedRealCurves(RenderPhrase phrase) => new();
        public UExpressionDescriptor[] GetSuggestedExpressions(USinger singer, URenderSettings renderSettings) => new[] {
            new UExpressionDescriptor("GYU style: 0 neutral, 1 soft, 2 breathy, 3 energetic, 4 relative C, 5 relative B", StyleCurve, 0, 5, 0) { type = UExpressionType.Curve },
        };
        public override string ToString() => Renderers.GYU_SINGER;
    }

    internal sealed class GyuRequest {
        public string protocol = "";
        public string source = "";
        public string language = "";
        public double tempo;
        public int sample_rate;
        public List<GyuNote> notes = new();
        public List<GyuPhoneme> phonemes = new();
        public Dictionary<string, List<GyuPoint>> curves = new();
        public Dictionary<string, object> style = new();
    }
    internal sealed class GyuNote { public float pitch; public double start; public double duration; public string lyric = ""; }
    internal sealed class GyuPhoneme { public string phoneme = ""; public double start; public double duration; }
    internal sealed class GyuPoint { public double time; public double value; }
}
