from ipdb import set_trace
import os

from vidbase.vidplayer import get_video_infos


def get_video_dimensions(input_video):
    input_video_info = get_video_infos(input_video)
    width, height = input_video_info['img_size']
    nr_frames = input_video_info['duration'] * input_video_info['fps']
    return width, height, nr_frames


def rescale(input_video, max_width, **kwargs):
    """ Rescales a video clip by a factor of 2 ** k, such that the width is
    smaller than max_width.

    Inputs
    ------
    in_video: str
        Path to input video.

    max_width: int
        Maximum width of the rescaled video.

    width, height, nr_frames: int, default None
        Width, height and number of frames of the original video. If None they
        will be computed.

    thresh: int, default 10
        The maximum difference allowed between the number of frames of the 
        original movie and the nuber of frames of the rescaled video.

    output_video: str, default '/dev/shm/<video_name>'
        Path to output video.

    get_dimensions: boolean, default False
        If True, outputs also the new dimesions of the rescaled video.

    Output
    ------
    status: str
        The status of the rescale:
            'sym_link': the video was small enough; the output is just a
            symbolic link.
            'rescaled': the video was succesfully rescaled.
            'bad_encoding': the input and output video don't have the same
            number of frames.

    output_video: str
        Path to output video.


    """
    # Get width, hight, thresh
    width = kwargs.get('width', None)
    height = kwargs.get('height', None)
    nr_frames = kwargs.get('nr_frames', None)
    thresh = kwargs.get('thresh', 10)
    get_dimensions = kwargs.get('get_dimensions', False)
    output_video = kwargs.get('output_video', os.path.join(
        '/dev/shm/', input_video.split('/')[-1]))

    try:
        if width is None or height is None or nr_frames is None:
            input_video_info = get_video_infos(input_video)
            width, height = input_video_info['img_size']
            nr_frames = input_video_info['duration'] * input_video_info['fps']
    except AssertionError:
        if get_dimension:
            return 'bad_encoding', '', (-1, -1, -1)
        else:
            return 'bad_encoding', ''

    # Find the appropriate scaling ratio.
    tt = 1
    while not width / tt <= max_width:
        tt *= 2
    new_width = width / tt
    new_height = height / tt

    if tt == 1:
        # No resizing needed; video is originally small enough.
        #os.system('ln -s %s %s' % (input_video, output_video))
        output_video = input_video
        status = 'sym_link'
    else:
        # Resize movie clip.
        #os.system('ffmpeg -i %s -s %dx%d %s' % (
        #   input_video, new_width, new_height, output_video))

        # Use mencoder to resize videos; it is more reliable. Thanks to
        # Danila for the suggestion.
        os.system(('mencoder %s -vf scale=%d:%d ' +
                   '-nosound -ovc lavc -o %s') %   # -oac lavc -ac copy
                   (input_video, new_width, new_height, output_video))
        status = 'rescaled'

        # Check the time is almost the same.
        output_video_info = get_video_infos(output_video)
        if abs(output_video_info['duration'] * output_video_info['fps']
               - nr_frames) > thresh:
            status = 'bad_encoding'

    #print 'Original video: ' + input_video_info.__str__()
    #print 'Rescaled video: ' + output_video_info.__str__()
    if get_dimensions:
        return status, output_video, (new_width, new_height, nr_frames)
    else:
        return status, output_video
