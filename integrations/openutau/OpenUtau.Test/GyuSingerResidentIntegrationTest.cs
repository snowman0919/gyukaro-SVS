using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using NAudio.Wave;
using OpenUtau.Core;
using OpenUtau.Core.GyuSinger;
using OpenUtau.Core.Render;
using Xunit;

namespace OpenUtau.Test {
    public class GyuSingerResidentIntegrationTest {
        [Fact]
        public async Task NativeRendererReturnsResidentPhraseAudio() {
            Assert.False(string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("GYU_RENDERER_URL")));
            DocManager.Inst.Initialize(Thread.CurrentThread, TaskScheduler.Default);
            var suffix = string.Concat(BitConverter.GetBytes(DateTime.UtcNow.Ticks).Select(value => (char)('\uac00' + value)));
            var phrase = GyuSingerRendererTest.Phrase(secondLyric: suffix);
            var renderer = new GyuSingerRenderer();
            Directory.CreateDirectory(PathManager.Inst.CachePath);
            var before = Directory.EnumerateFiles(PathManager.Inst.CachePath, "gyu-*.wav").ToHashSet();
            var result = await renderer.Render(phrase, new Progress(phrase.phones.Length), 0, new CancellationTokenSource());
            Assert.NotNull(result.samples);
            Assert.InRange(result.samples.Length, 44000, 44200);
            Assert.Contains(result.samples, sample => Math.Abs(sample) > 0.001f);
            var afterFirst = Directory.EnumerateFiles(PathManager.Inst.CachePath, "gyu-*.wav").ToHashSet();
            Assert.Single(afterFirst.Except(before));
            var cached = await renderer.Render(phrase, new Progress(phrase.phones.Length), 0, new CancellationTokenSource());
            Assert.Equal(afterFirst, Directory.EnumerateFiles(PathManager.Inst.CachePath, "gyu-*.wav").ToHashSet());
            Assert.Equal(result.samples, cached.samples);
            var editedPhrase = GyuSingerRendererTest.Phrase(secondTone: 66, secondLyric: suffix);
            var edited = await renderer.Render(editedPhrase, new Progress(editedPhrase.phones.Length), 0, new CancellationTokenSource());
            var afterEdit = Directory.EnumerateFiles(PathManager.Inst.CachePath, "gyu-*.wav").ToHashSet();
            Assert.Single(afterEdit.Except(afterFirst));
            Assert.NotEqual(result.samples, edited.samples);

            var output = Environment.GetEnvironmentVariable("GYU_BEHAVIOR_OUTPUT");
            if (!string.IsNullOrWhiteSpace(output)) {
                Directory.CreateDirectory(output);
                var cases = new[] {
                    ("ko", GyuSingerRendererTest.Phrase(firstLyric: "하늘빛 노래", secondLyric: "불러요", noteDuration: 960)),
                    ("en", GyuSingerRendererTest.Phrase(firstLyric: "sing to the sky", secondLyric: "follow the light", noteDuration: 960)),
                    ("ja", GyuSingerRendererTest.Phrase(firstLyric: "空へ歌おう", secondLyric: "光を追う", noteDuration: 960)),
                    ("note_pitch_edit", GyuSingerRendererTest.Phrase(firstTone: 62, secondTone: 66, firstLyric: "하늘빛 노래", secondLyric: "불러요", noteDuration: 960)),
                    ("lyric_edit", GyuSingerRendererTest.Phrase(firstLyric: "사랑을 담아서", secondLyric: "마음을 전해요", noteDuration: 960)),
                    ("user_pitch_edit", GyuSingerRendererTest.Phrase(firstLyric: "하늘빛 노래", secondLyric: "불러요", pitchDeviation: 100, noteDuration: 960)),
                    ("style_energetic", GyuSingerRendererTest.Phrase(firstLyric: "하늘빛 노래", secondLyric: "불러요", style: 3, noteDuration: 960)),
                };
                foreach (var item in cases) {
                    var rendered = await renderer.Render(item.Item2, new Progress(item.Item2.phones.Length), 0, new CancellationTokenSource());
                    using var writer = new WaveFileWriter(Path.Combine(output, item.Item1 + ".wav"), WaveFormat.CreateIeeeFloatWaveFormat(44100, 1));
                    foreach (var sample in rendered.samples) writer.WriteSample(sample);
                }
            }
        }
    }
}
