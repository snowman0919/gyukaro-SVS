using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
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
        }
    }
}
