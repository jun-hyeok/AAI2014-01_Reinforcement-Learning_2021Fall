import cmd
import csv
import datetime
import os

import pandas as pd

complete_path = (
    lambda curdir, item: item if os.path.isfile(curdir + item) else item + "/"
)


def print_error(e: Exception):
    print(f"[{e.__class__.__name__}] {e.strerror}: '{e.filename}'")


class ConfigShell(cmd.Cmd):
    def __init__(self, file="") -> None:
        """Initialize the ConfigShell class.

        Args:
            file (str): path to the configuration file
        Errors:
            FileNotFoundError: The file is invalid or does not exist.
            pandas.errors.EmptyDataError: The file is empty.
            pandas.errors.ParserError: Parameter keys must be in row 2(header=1) of the file.
            IndexError: There is no recent log in the file.
        """
        super().__init__()

        # [SET display setting]
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 100)

        self.intro = "Start Configuration Shell"
        self.prompt = "(Configuration)>>>"

        # [SET configuration parameter]
        self._write_params = True
        self._add_params = False
        # self.use_default = False
        self.parameter = dict()

        # [CONFIGURE file]
        self.insh = InnerShell()
        self.do_read(file)
        self.cmdloop()

    def do_read(self, path: str):
        """Read the configuration file.

        Prompt:
            (Configuration)>>>read [path]

        Errors:
            FileNotFoundError: The file is invalid or does not exist.
            pandas.errors.EmptyDataError: The file is empty.
            pandas.errors.ParserError: Parameter keys must be in row 2(header=1) of the file.
            IndexError: There is no recent log in the file.
        """

        file = path.split()[0] if path else None
        while True:
            if not file:
                self.abort()
            else:
                try:
                    # FileNotFoundError, pandas.errors.EmptyDataError, pandas.errors.ParserError
                    self.config = pd.read_csv(file, header=1)
                    self.file = file
                    # IndexError
                    recent = self.config.iloc[-1].to_dict()
                    self.parameter.update(recent)
                    print(f"Config file : {os.path.abspath(self.file)}")
                    break
                except FileNotFoundError as e:
                    print_error(e)
                    alert = """\
\a⚠️ Invalid configuration file path!
===============================================================================
Next time, you can enter it like this on the declaration in your script.
Example: ... = ConfigShell(..., file="/path/to/the/file")
        or
         ... = ConfigShell("/path/to/the/file", ...)
===============================================================================
Enter the path to the configuration file(.csv) to continue or Enter to abort.
                    """
                    print(alert)
                    self.insh.cmdloop(cmd="read", prompt="file: ")
                    file = self.insh.file
                except Exception as e:
                    print_error(e)
                    print("\a⚠️ Check configuration file!")
                    file = None

    def complete_read(self, text, line, begidx, endidx):
        # get current directory
        try:
            _, curdir = line[:begidx].split()
        except:
            curdir = "./"
        curdir = os.path.expanduser(curdir)

        # get list of files in current directory
        import string

        items = os.listdir(curdir)
        ignore_cond = string.punctuation
        ignore = lambda f: not any(f.startswith(i) for i in ignore_cond)
        filtered_items = filter(ignore, items)

        return [
            complete_path(curdir, item)
            for item in filtered_items
            if (item.startswith(text) if text else True)
        ]

    # type casting is need in main script!!
    def do_show(self, params=None):
        """Show the configuration parameters.

        Prompt:
            (Configuration)>>>show
            (Configuration)>>>show [parameter]
        """
        if params:
            for param in params.split():
                if param in self.parameter.keys():
                    print(f"{param:>30} = {self.parameter[param]}")
        else:
            for key, value in self.parameter.items():
                print(f"{key:>30} = {value}")

    def do_set(self, kwargs: str):
        """Set the configuration parameters.

        Prompt:
            (Configuration)>>>set [key] [value]
        """
        key, *value = kwargs.split()
        if key in self.parameter.keys():
            self.parameter[key] = value[0] if value[0] != "None" else ""
            print(key, self.parameter[key])
        else:
            print(f"unknown parameter: {key}")
            print("if you want to add the parameter, use `add` command")
        if len(value) > 2:
            del value[0]
            self.do_set(" ".join(value))

    def do_del(self, arg: str):
        """Delete the configuration parameters.

        Prompt:
            (Configuration)>>>del [key]
        """
        key = arg.split()[0]
        if key in self.parameter.keys():
            del self.parameter[key]
        else:
            print("unknown parameter")

    def completedefault(self, text, line, begidx, endidx):
        params = self.parameter.keys()
        if text:
            return [param for param in params if param.startswith(text)]
        else:
            return list(params)

    def do_add(self, arg: str):
        """Add the configuration parameters.

        Prompt:
            (Configuration)>>>add [key] [value]
        """
        key, *value = arg.split()
        print(key, value[0])
        if len(value) > 2:
            self.do_add(" ".join(value[1:]))
        else:
            self.add_params = input("Add these parameters?(Y/n):")
        if self.add_params in ["y", "Y"]:
            self.parameter[key] = value[0]
        else:
            print("The parameters are not added")

    def do_show_csv(self, arg: str):
        """Show the configuration file.

        Prompt:
            (Configuration)>>>show_csv
        """
        print(self.config.tail())

    def do_done(self, arg):
        """Exit the configuration shell and Start the main script.

        Prompt:
            (Configuration)>>>done
        Shortcut:
            Ctrl+D
        """
        print("The configuration is all set.")
        return True

    def abort(self):
        """Exit the configuration shell and Exit the program.

        Prompt:
            (Configuration)>>>abort
            (Configuration)>>>exit
        """
        raise SystemExit("Configuration failed. Abort Experiment")

    def do_write(self, arg: str):
        """Write the configuration file or not.

        Prompt:
            (Configuration)>>>write ["false", "False", "f", "F", "0"]
        """
        flag = arg.split()[0]
        if flag in ["false", "False", "f", "F", "0"]:
            self._write_params = False

    do_EOF = do_done
    do_exit = abort

    def write_parameters(self):
        """Write the configuration file."""
        with open(self.file, "a+", newline="") as file:
            csv_dict_writer = csv.DictWriter(file, fieldnames=list(self.config.keys()))
            csv_dict_writer.writerow(self.parameter)

    def get_path(self, root: str):
        """Get the path of the file.

        Args:
            root: The root path of the file.

        Returns:
            parametors: The configuration parameters.
            env: The path of the env file.
            logdir: The path of the log directory.
            savedir: The path of the save directory.
        """
        try:
            env_ver = str(self.parameter.get("env", 1.0))
            tag = self.parameter.get("tag", "")
            load_model = self.parameter.get("load_model", "")
            load_model = "" if pd.isna(load_model) else load_model
            self.parameter["load_model"] = load_model
        except Exception as e:
            print_error(e)
            raise SystemExit("The parameters are invalid. Abort Experiment")

        date = datetime.datetime.now().strftime("%Y%m%d")
        time = datetime.datetime.now().strftime("%H-%M-%S")

        env = os.path.join(root, "envs", env_ver, "build")
        if not os.path.exists(env + ".x86_64"):
            raise SystemExit("The environment is invalid. Abort Experiment")

        if not load_model:
            logdir = os.path.join(root, "logs", env_ver, "train", date, time, tag)
        else:
            if "ckpt" in load_model:
                _, loads = str.split(load_model, "saves")
                logdir = os.path.join(root, "logs", env_ver, "test", loads, date, time, tag)  # type: ignore
            else:
                print(f"{load_model}")
                raise SystemExit("The path is invalid. Abort Experiment")

        savedir = os.path.join(root, "saves", env_ver, date, time, tag)

        if self._write_params:
            self.parameter["key"] = f"{date} {time}"
            self.write_parameters()
        return self.parameter, env, logdir, savedir


class InnerShell(cmd.Cmd):
    def __init__(self, *args, **kwargs):
        """Inner shell for the configuration file."""
        super().__init__()

    def abort(self):
        raise SystemExit("Configuration failed. Abort Experiment")

    def completedefault(self, text, line, begidx, endidx):
        line = self.cmd + " " + line
        begidx += len(self.cmd) + 1
        return getattr(ConfigShell, "complete_" + self.cmd)(
            self, text, line, begidx, endidx
        )

    completenames = completedefault

    def default(self, line):
        self.file = line
        return True

    def cmdloop(self, intro=None, cmd=None, prompt=None):
        self.cmd = cmd
        self.prompt = prompt
        super().cmdloop(intro=intro)

    do_EOF = abort
    emptyline = abort


if __name__ == "__main__":
    configsh = ConfigShell("project/config.csv")
    configsh.get_path("project")
