using System;
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
            var phrase = GyuSingerRendererTest.Phrase();
            var renderer = new GyuSingerRenderer();
            var result = await renderer.Render(phrase, new Progress(phrase.phones.Length), 0, new CancellationTokenSource());
            Assert.NotNull(result.samples);
            Assert.InRange(result.samples.Length, 44000, 44200);
            Assert.Contains(result.samples, sample => Math.Abs(sample) > 0.001f);
        }
    }
}
