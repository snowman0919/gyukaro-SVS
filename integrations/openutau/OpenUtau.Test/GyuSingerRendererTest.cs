using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using OpenUtau.Core;
using OpenUtau.Core.Format;
using OpenUtau.Core.GyuSinger;
using OpenUtau.Core.Render;
using OpenUtau.Core.Ustx;
using Xunit;

namespace OpenUtau.Test {
    public class GyuSingerRendererTest {
        sealed class FakeSinger : USinger {
            public FakeSinger() { found = loaded = true; }
            public override string Id => "gyu-test";
            public override string Name => "GYU Test";
            public override USingerType SingerType => USingerType.GyuSinger;
            public override IList<USubbank> Subbanks => Array.Empty<USubbank>();
            public override bool TryGetOto(string phoneme, out UOto oto) { oto = UOto.OfDummy(phoneme); return true; }
        }

        internal static RenderPhrase Phrase(int firstTone = 60, int secondTone = 64, string firstLyric = "하", string secondLyric = "늘", int pitchDeviation = 0, int style = 0, int noteDuration = 480) {
            var project = Ustx.Create();
            var styleDescriptor = new UExpressionDescriptor("GYU style", "gyus", 0, 5, 0) { type = UExpressionType.Curve };
            project.RegisterExpression(styleDescriptor);
            var track = project.tracks[0];
            track.Singer = new FakeSinger();
            track.RendererSettings.renderer = Renderers.GYU_SINGER;
            var part = new UVoicePart { duration = noteDuration * 2 };
            var first = project.CreateNote(firstTone, 0, noteDuration); first.lyric = firstLyric;
            var second = project.CreateNote(secondTone, noteDuration, noteDuration); second.lyric = secondLyric;
            part.notes.Add(first); part.notes.Add(second);
            part.phonemes.Add(new UPhoneme { rawPosition = 0, rawPhoneme = "ha", Parent = first });
            part.phonemes.Add(new UPhoneme { rawPosition = noteDuration, rawPhoneme = "nul", Parent = second });
            if (pitchDeviation != 0) {
                var curve = new UCurve(project.expressions[Ustx.PITD]);
                curve.xs.AddRange(new[] { 0, noteDuration * 2 }); curve.ys.AddRange(new[] { pitchDeviation, pitchDeviation });
                part.curves.Add(curve);
            }
            if (style != 0) {
                var curve = new UCurve(styleDescriptor);
                curve.xs.AddRange(new[] { 0, noteDuration * 2 }); curve.ys.AddRange(new[] { style, style });
                part.curves.Add(curve);
            }
            project.parts.Add(part);
            project.Validate(new ValidateOptions { SkipPhonemizer = true });
            return RenderPhrase.FromPart(project, track, part).Single();
        }

        [Fact]
        public void MultiNotePhraseMapsScoreLyricsAndLanguage() {
            var request = GyuSingerRenderer.BuildRequest(Phrase());
            Assert.Equal(2, request.notes.Count);
            Assert.Equal(new[] { "하", "늘" }, request.notes.Select(note => note.lyric));
            Assert.Equal("ko", request.language);
        }

        [Fact]
        public void NoteAndLyricEditsChangePayload() {
            var original = GyuSingerRenderer.BuildRequest(Phrase());
            var edited = GyuSingerRenderer.BuildRequest(Phrase(secondTone: 67, secondLyric: "빛"));
            Assert.NotEqual(original.notes[1].pitch, edited.notes[1].pitch);
            Assert.NotEqual(original.notes[1].lyric, edited.notes[1].lyric);
        }

        [Fact]
        public void UserPitchAndStyleCurvesRemainAuthoritative() {
            var original = GyuSingerRenderer.BuildRequest(Phrase());
            var edited = GyuSingerRenderer.BuildRequest(Phrase(pitchDeviation: 100, style: 3));
            Assert.True(edited.curves["pitch"].Average(point => point.value) > original.curves["pitch"].Average(point => point.value) + 0.5);
            Assert.Equal("energetic", edited.style["preset"]);
        }

        [Theory]
        [InlineData("노래", "ko")]
        [InlineData("sing", "en")]
        [InlineData("歌う", "ja")]
        public void DetectsSupportedLanguage(string lyric, string language) {
            Assert.Equal(language, GyuSingerRenderer.DetectLanguage(lyric));
        }

        [Fact]
        public void ExampleProjectLoadsAllNativeGyuTracks() {
            SingerManager.Inst.Initialize();
            var project = Ustx.Load(Path.Combine("Files", "openutau_v09.ustx"));
            Assert.Equal(3, project.tracks.Count);
            Assert.Equal(3, project.parts.OfType<UVoicePart>().Count());
            Assert.All(project.tracks, track => {
                Assert.Equal(USingerType.GyuSinger, track.Singer.SingerType);
                Assert.Equal(Renderers.GYU_SINGER, track.RendererSettings.renderer);
                Assert.IsType<GyuSingerRenderer>(track.RendererSettings.Renderer);
            });
            Assert.Equal(new[] { "GYU KO", "GYU EN", "GYU JA" }, project.tracks.Select(track => track.TrackName));
        }
    }
}
