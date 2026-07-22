from anc_audio_analysis import (
    analyze_anc_audio,
    print_analysis_results,
)

results = analyze_anc_audio(
    audio_path="anc_recording.wav",
    anc_off_range=(0.0, 5.0),
    anc_on_range=(6.0, 11.0),
    n_fft=8192,
    hop_length=2048,
    window_type="hann",
    channel=None,
    mono=True,
    plot=True,
    show_plot=False,
    plot_path="anc_result.png",
    octave_plot_path="anc_on_off_octave.png",

)

print_analysis_results(results)