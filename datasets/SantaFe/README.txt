http://www.physionet.org/physiobank/database/santa-fe/

This is a multivariate data set recorded from a patient in the sleep laboratory of the Beth Israel Hospital (now the Beth Israel Deaconess Medical Center) in Boston, Massachusetts. This data set was extracted from record slp60 Click to view waveforms of the MIT-BIH Polysomnographic Database, and it was submitted to the Santa Fe Time Series Competition in 1991 by our group. The data are presented in text form and have been split into two sequential parts, b1.txt and b2.txt. Each line contains simultaneous samples of three parameters; the interval between samples in successive lines is 0.5 seconds. The first column is the heart rate, the second is the chest volume (respiration force), and the third is the blood oxygen concentration (measured by ear oximetry). The sampling frequency for each measurement is 2 Hz (i.e., the time interval between measurements in successive rows is 0.5 seconds).

The heart rate was determined by locating the QRS complexes using an automated beat detector, measuring the RR intervals (the time intervals between successive QRS complexes in the electrocardiogram), taking the reciprocals, and then converting the series of reciprocals to a series with samples at equal time intervals by interpolation using tach. There were no abnormal beats (sudden changes in the heart rate are not artifacts).

The respiration and blood oxygen data are given in uncalibrated analog-to-digital converter units. These two sensors slowly drift with time (and are therefore occasionally rescaled by a technician) and can be detached by the motion of the patient, hence their calibration is not constant over the data set. These signals were originally sampled at 250 Hz; the 2 Hz samples given here were derived by summing 20 samples (80 milliseconds) of each original signal in windows centered on the times corresponding to the heart rate samples. (The description written in 1991 to accompany this data set stated erroneously that these samples were averages rather than sums. The method used for summation was not recorded, but it apparently included additional steps since sums of 20 samples from the original record do not exactly match those recorded in these files. Thanks to Takayoshi Shiraki of the University of Tokyo for reporting that the samples are not averages and that they closely approximate sums.)

Between roughly 4 hours 30 minutes and 4 hours 34 minutes from the start of the file, the sensors were disconnected. The following table gives the times and stages of sleep, as determined by a neurologist looking at the EEG (W = awake, 1 and 2 = waking/sleep stages, R = REM sleep):

   0:00: W,    2:00: 1,    2:30: W,    3:30: 1,    9:30: W,
  10:00: 1,   11:00: W,   12:00: 1,   15:30: 2,   16:00: 1,
  36:30: W,   38:00: 1,   39:30: 2,   43:00: 1,   44:30: 2,
  45:00: W,   45:30: 1,   46:30: W,   47:30: 2,   48:00: 1,
  49:00: 2,   50:30: 1,   51:00: 2,   51:30: 1,   52:00: 2,
  52:30: W,   53:00: 1,   53:30: W,   54:00: 1,   55:30: 2,
  56:30: W, 1:20:00: 1, 1:20:30: W, 1:22:00: 1, 1:23:00: W,
1:26:00: 1, 1:27:00: W, 1:29:30: 1, 1:30:30: W, 1:31:00: 1,
1:31:30: W, 1:32:00: 1, 1:34:00: W, 1:40:00: 1, 1:41:00: W,
1:41:30: 1, 1:43:00: 2, 1:43:30: 1, 1:45:00: 2, 1:51:30: R,
2:05:30: W, 2:44:00: 1, 2:48:00: W, 2:49:00: 1, 2:51:00: W,
2:58:30: 1, 2:59:30: W, 3:00:00: 1, 3:01:00: W, 3:01:30: 1,
3:02:00: W, 3:07:00: 1, 3:08:30: W, 3:12:00: 1, 3:13:00: W,
3:13:30: 1, 3:19:00: 2, 3:19:30: W, 3:22:00: 1, 3:22:30: W,
3:23:00: 1, 3:38:00: W, 3:39:00: 1, 3:39:30: W, 3:40:00: 1,
3:41:00: W, 3:41:30: 1, 3:44:30: W, 4:13:00: 1, 4:14:00: W,
4:17:00: 1, 4:26:30: W, 4:28:30: 1, 4:29:00: W, 4:29:30: 1,
4:45:00: 2, 4:45:30: 1, 4:46:00: 2, 4:46:30: 1, 4:48:30: 2,
4:49:00: 1, 4:50:00: 2, 4:50:30: 1, 4:51:00: 2, 4:51:30, 1,
4:53:30: 2, 4:54:00: 1, 4:55:30: 2, 4:56:00: 1, 5:00:00, 2,
5:00:30: 1, 5:13:00: R, 5:13:30: 1

 This patient shows sleep apnea (periods during which he takes a few quick breaths and then stops breathing for up to 45 seconds). Sleep apnea is medically important because it leads to sleep deprivation and occasionally death. There are three primary research questions associated with this data set:

    Can part of the temporal variation in the heart rate be explained by a low-dimensional mechanism, or is it due to noise or external inputs?
    How do the evolution of the heart rate, the respiration rate, and the blood oxygen concentration affect each other? (A correlation between breathing and the heart rate, called respiratory sinus arrhythmia, is almost always observed.)
    Can the episodes of sleep apnea (stoppage of breathing) be predicted from the preceding data?


Reason for choice:

    Heart rate variability
    There is growing (but still controversial) evidence that the observed variations in the heart rate might be related to a low-dimensional governing mechanism; understanding this mechanism is obviously very important in order to understand its failures (i.e., heart attacks).
    Multi-dimensional data sets
    These data provide simultaneous measurements of a number of potentially interacting variables; it is an open question how best to use the extra information to learn about how the variables interact. Most importantly, there is interest in verifying and understanding the coupling between respiration and the heart rate.
    Non-stationary data
    These data were recorded with as much care as is possible, but the experimental system (the sleeping patient) is obviously non-stationary. A successful analysis of these data must attempt to distinguish the presumed internal dynamics from changes in the patient&rsdquo;s state.
