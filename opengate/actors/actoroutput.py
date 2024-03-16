from ..exception import warning, fatal
from ..base import GateObject
from pathlib import Path
from ..image import (
    write_itk_image,
    update_image_py_to_cpp,
    create_3d_image,
    get_py_image_from_cpp_image,
    sum_itk_images,
    divide_itk_images,
)
from ..utility import ensure_filename_is_str


class DataItemBase:
    _tuple_length = 0

    def __init__(self, *args, data=None, **kwargs):
        if data is None:
            data = [None] * self._tuple_length
        self.set_data(data)

    def __add__(self, other):
        return NotImplemented

    def __iadd__(self, other):
        return NotImplemented

    def __truediv__(self, other):
        return NotImplemented

    def write(self, *args, **kwargs):
        raise NotImplementedError(f"This is the base class. ")

    @property
    def data_is_none(self):
        return any([d is None for d in self.data])

    def set_data_items(self, *data_items):
        self.set_data(data_items)

    def set_data(self, data):
        if len(data) != self._tuple_length:
            raise ValueError
        else:
            try:
                self.data = tuple(data)
            except TypeError:
                raise TypeError(
                    f"Incompatible argument data: {data}. Must be an iterable. "
                )

    def call_method_on_data(self, method_name, *args, **kwargs):
        for d in self.data:
            getattr(d, method_name)(d, *args, **kwargs)


class SingleDataItem(DataItemBase):
    _tuple_length = 1


class DoubleDataItem(DataItemBase):
    _tuple_length = 2

    def __iadd__(self, other):
        self.set_data(
            [self.data[i].__iadd__(other.data[i]) for i in range(self._tuple_length)]
        )
        return self

    def __add__(self, other):
        return type(self)(
            data=[self.data[i] + other.data[i] for i in range(self._tuple_length)]
        )

    def write(self, path, suffix=(None, None)):
        for d, s in zip(self.data):
            d.write(path, suffix=s)  # FIXME use suffix


class QuotientDataItem(DoubleDataItem):

    def __init__(self, *args, numerator=None, denominator=None, **kwargs):
        if numerator is not None and denominator is not None:
            kwargs["data"] = (numerator, denominator)
        super().__init__(*args, **kwargs)

    @property
    def numerator(self):
        return self.data[0]

    @property
    def denominator(self):
        return self.data[1]


class ImageDataItem(SingleDataItem):

    def __init__(self, *args, image=None, **kwargs):
        if image is not None:
            kwargs["data"] = (image,)
        super().__init__(*args, **kwargs)

    @property
    def image(self):
        return self.data[0]

    def __iadd__(self, other):
        if self.data_is_none:
            raise ValueError(
                "This data item does not contain any data yet. "
                "Use set_data() before applying any operations. "
            )
        self.set_data(sum_itk_images([self.image, other.image]))
        return self

    def __add__(self, other):
        if self.data_is_none:
            raise ValueError(
                "This data item does not contain any data yet. "
                "Use set_data() before applying any operations. "
            )
        return type(self)(image=sum_itk_images([self.image, other.image]))

    def __truediv__(self, other):
        if self.data_is_none:
            raise ValueError(
                "This data item does not contain any data yet. "
                "Use set_data() before applying any operations. "
            )
        return type(self)(image=divide_itk_images(self.image, other.image))

    def __itruediv__(self, other):
        if self.data_is_none:
            raise ValueError(
                "This data item does not contain any data yet. "
                "Use set_data() before applying any operations. "
            )
        self.set_data(divide_itk_images(self.image, other.image))
        return self

    def set_image_properties(self, spacing=None, origin=None):
        if not self.data_is_none:
            if spacing is not None:
                self.image.SetSpacing(spacing)
            if origin is not None:
                self.image.SetOrigin(origin)

    def create_empty_image(
        self, size, spacing, pixel_type="float", allocate=True, fill_value=0
    ):
        self.set_data(create_3d_image(size, spacing, pixel_type, allocate, fill_value))

    def write(self, path):
        write_itk_image(self.image, ensure_filename_is_str(path))


class QuotientImageDataItem(QuotientDataItem):

    def __init__(self, *args, numerator_image=None, denominator_image=None, **kwargs):
        super().__init__(
            *args,
            numerator=ImageDataItem(image=numerator_image),
            denominator=ImageDataItem(image=denominator_image),
            **kwargs,
        )

    def set_images(self, numerator_image, denominator_image):
        self.set_data(ImageDataItem(numerator_image), ImageDataItem(denominator_image))

    def create_empty_image(
        self, size, spacing, pixel_type="float", allocate=True, fill_value=0
    ):
        self.numerator.create_empty_image(
            size, spacing, pixel_type, allocate, fill_value
        )
        self.denominator.create_empty_image(
            size, spacing, pixel_type, allocate, fill_value
        )

    def set_image_properties(self, spacing=None, origin=None):
        self.numerator.set_image_properties(spacing, origin)
        self.denominator.set_image_properties(spacing, origin)


def _setter_hook_belongs_to(self, belongs_to):
    if belongs_to is None:
        fatal("The belongs_to attribute of an ActorOutput cannot be None.")
    try:
        belongs_to_name = belongs_to.name
    except AttributeError:
        belongs_to_name = belongs_to
    return belongs_to_name


def _setter_hook_path(self, path):
    return Path(path)


class ActorOutput(GateObject):
    user_info_defaults = {
        "belongs_to": (
            None,
            {
                "doc": "Name of the actor to which this output belongs.",
                "setter_hook": _setter_hook_belongs_to,
                "required": True,
            },
        ),
        "output_filename": (
            None,
            {
                "doc": "Filename for the data represented by this actor output. "
                "Relative paths and filenames are taken "
                "relative to the global simulation output folder "
                "set via the Simulation.output_path option. ",
            },
        ),
        "extra_suffix": (
            None,
            {
                "doc": "Extra suffix to be added to the filenames (before the run_index). ",
            },
        ),
        "write_to_disk": (
            True,
            {
                "doc": "Should the data be written to disk?",
            },
        ),
        "keep_data_in_memory": (
            True,
            {
                "doc": "Should the data be kept in memory after the end of the simulation? "
                "Otherwise, it is only stored on disk and needs to be re-loaded manually. "
                "Careful: Large data structures like a phase space need a lot of memory.",
            },
        ),
        "keep_data_per_run": (
            False,
            {
                "doc": "In case the simulation has multiple runs, should separate results per run be kept?"
            },
        ),
        "auto_merge": (
            True,
            {
                "doc": "In case the simulation has multiple runs, should results from separate runs be merged?"
            },
        ),
    }

    default_suffix = ""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.output_filename is None:
            self.output_filename = f"output_{self.name}_from_actor_{self.belongs_to_actor.name}.{self.default_suffix}"

        self.data_per_run = {}  # holds the data per run in memory
        self.merged_data = None  # holds the data merged from multiple runs in memory

    @property
    def data(self):
        if len(self.data_per_run) > 1:
            warning(
                f"You are using the convenience property 'data' to access the data in this actor output. "
                f"This returns you the data from the first run, but the actor output stores "
                f"data from {len(self.data_per_run)} runs. "
                f"To access them, use 'data_per_run[RUN_INDEX]' instead or 'merged_data'. "
            )
        return self.data_per_run[0]

    @property
    def belongs_to_actor(self):
        return self.simulation.actor_manager.get_actor(self.belongs_to)

    def merge_data(self, list_of_data):
        raise NotImplementedError(
            f"Your are calling this method from the base class {type(self).__name__}, "
            f"but it should be implemented in the specific derived class"
        )

    def merge_data_from_runs(self):
        self.merged_data = self.merge_data(list(self.data_per_run.values()))

    def merge_into_merged_data(self, data):
        self.merged_data = self.merge_data([self.merged_data, data])

    def end_of_run(self, run_index):
        if self.keep_data_per_run is False:
            if self.auto_merge is True:
                self.merge_into_merged_data(self.data_per_run[run_index])
            self.data_per_run[run_index] = None

    def end_of_simulation(self):
        if self.auto_merge is True:
            self.merge_data_from_runs()
        if self.keep_data_per_run is False:
            for k in self.data_per_run:
                self.data_per_run[k] = None

    def store_data(self, data, run_index):
        self.data_per_run[run_index] = data

    def load_data(self, which):
        raise NotImplementedError(
            f"Your are calling this method from the base class {type(self).__name__}, "
            f"but it should be implemented in the specific derived class"
        )

    def collect_data(self, which, return_identifier=False):
        if which == "merged":
            data = [self.merged_data]
            identifiers = ["merged"]
        elif which == "all_runs":
            data = list(self.data_per_run.values())
            identifiers = list(self.data_per_run.keys())
        elif which == "all":
            data = list(self.data_per_run.values())
            data.append(self.merged_data)
            identifiers = list(self.data_per_run.keys())
            identifiers.append("merged")
        else:
            try:
                ri = int(which)
            except ValueError:
                fatal(f"Invalid argument which in method collect_images(): {which}")
            data = [self.data_per_run[ri]]
            identifiers = [ri]
        if return_identifier is True:
            return data, identifiers
        else:
            return data

    def write_data(self, which):
        if which == "all_runs":
            for i, data in self.data_per_run.items():
                if data is not None:
                    self._write_data(data, self.get_output_path(i))
        elif which == "merged":
            self._write_data(self.merged_data, self.get_output_path(which))
        elif which == "all":
            self.write_data("all_runs")
            self.write_data("merged")
        else:
            try:
                data = self.data_per_run[which]
            except KeyError:
                fatal(
                    f"Invalid argument 'which' in method write_data(): {which}. "
                    f"Allowed values are 'all', 'all_runs', 'merged', or a valid run_index"
                )
            self._write_data(data, self.get_output_path(which))

    def _write_data(self, data, path):
        """A concrete class must implement this method.
        The argument 'path' is a Path object from the pathlib library.
        The concrete _write_path() method should NOT alter the path!
        """
        raise NotImplementedError(
            f"Your are calling this method from the base class {type(self).__name__}, "
            f"but it should be implemented in the specific derived class"
        )

    def write_data_if_requested(self, *args, **kwargs):
        if self.write_to_disk is True:
            self.write_data(*args, **kwargs)

    def get_output_path(self, which):
        full_data_path = self.simulation.get_output_path(self.output_filename)
        if self.extra_suffix is not None:
            full_data_path = full_data_path.with_name(
                full_data_path.stem + f"_{self.extra_suffix}" + full_data_path.suffix
            )
        if which == "merged":
            return full_data_path.with_name(
                full_data_path.stem + f"_merged" + full_data_path.suffix
            )
        else:
            try:
                run_index = int(which)
            except ValueError:
                fatal(
                    f"Invalid argument 'which' in get_output_path() method "
                    f"of {type(self).__name__} called {self.name}"
                    f"Valid arguments are a run index (int) or the term 'merged'. "
                )
            return full_data_path.with_name(
                full_data_path.stem + f"_run{run_index:04f}" + full_data_path.suffix
            )

    def close(self):
        if self.keep_data_in_memory is False:
            self.data_per_run = {}
            self.merged_data = None
        super().close()


class ActorOutputImage(ActorOutput):
    user_info_defaults = {
        "merge_method": (
            "sum",
            {
                "doc": "How should images from runs be merged?",
                "allowed_values": ("sum",),
            },
        ),
        "size": (
            None,
            {
                "doc": "Size of the image in voxels.",
            },
        ),
        "spacing": (
            None,
            {
                "doc": "Spacing of the image.",
            },
        ),
    }

    default_suffix = "mhd"

    # override method
    def merge_data(self, list_of_data):
        if self.merge_method == "sum":
            return sum_itk_images(list_of_data)

    # override method
    def _write_data(self, image, path):
        """This 'private' method should not be called from outside,
        but only via the write_data() method (no underscore) from the base class
        """
        write_itk_image(image, ensure_filename_is_str(path))

    def set_image_properties(self, which, spacing=None, origin=None):
        images = self.collect_data(which)
        for im in images:
            if im is not None:
                if spacing is not None:
                    im.SetSpacing(spacing)
                if origin is not None:
                    im.SetOrigin(origin)

    def create_empty_image(
        self, run_index, pixel_type="float", allocate=True, fill_value=0
    ):
        self.data_per_run[run_index].call_method_on_data(
            "create_empty_image",
            self.size,
            self.spacing,
            pixel_type,
            allocate,
            fill_value,
        )

    def update_to_cpp_image(self, cpp_image, run_index, copy_data=False):
        update_image_py_to_cpp(
            self.data_per_run[run_index], cpp_image, copy_data=copy_data
        )

    def update_from_cpp_image(self, cpp_image, run_index):
        self.data_per_run[run_index] = get_py_image_from_cpp_image(cpp_image)


class ActorOutputRoot(ActorOutput):
    user_info_defaults = {
        "merge_method": (
            "append",
            {
                "doc": "How should images from runs be merged?",
                "allowed_values": ("append",),
            },
        ),
    }

    default_suffix = "root"

    def merge_data(self, list_of_data):
        if self.merge_method == "append":
            raise NotImplementedError("Appending ROOT files not yet implemented.")


actor_output_classes = {"root": ActorOutputRoot, "image": ActorOutputImage}
