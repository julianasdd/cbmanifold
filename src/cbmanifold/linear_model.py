from dataclasses import dataclass
import numpy as np


@dataclass
class LinearModel:
    label: str
    rate: np.ndarray = None
    drate: np.ndarray = None
    params0: np.ndarray = None

    is_dim_reduced: bool = False

    def predict(self, params):
        """
        Predict the rate matrix from the parameters.
        """

        if isinstance(params, (list, int, float)):
            params = np.array(params)
        assert (
            params.shape[-1] == self.drate.shape[0]
        ), f"Number of parameters ({params.shape[-1]}) must match number of parameter derivatives ({self.drate.shape[0]})"
        return self.rate + (params - self.params0) @ self.drate

    @classmethod
    def from_dict(cls, d):
        """
        Convert a dictionary to a LinearModel object.

        Parameters
        ----------
        d : dict
            Dictionary containing linear model data. Must have 'label' key.
            Optional keys are 'rate', 'drate', 'params0', 'is_dim_reduced'.
            Any additional keys will be added as attributes to the object.

        Returns
        -------
        LinearModel
            New LinearModel object with data from dictionary
        """
        # Create base LinearModel with required/optional fields
        lm = cls(
            label=d['label'],
            rate=d.get('rate'),
            drate=d.get('drate'),
            params0=d.get('params0'),
            is_dim_reduced=d.get('is_dim_reduced', False)
        )

        # Add any additional fields from dict as attributes
        for k, v in d.items():
            if k not in ['label', 'rate', 'drate', 'params0', 'is_dim_reduced']:
                setattr(lm, k, v)

        return lm


def generate_linear_model_old(
    label, rate_matrix, params, params0, target_dist=10.0, output_type="dict"
):
    """
    Generate a linear model of the rate matrix.

    Parameters:
    - label (str): Label for the linear model
    - rate_matrix (ndarray): Rate matrix (Trial x Time)
    - params (ndarray): Parameters (peak velocity, duration, etc.) shape (Trial x n_params)
    - params0 (array-like): Grand average parameters, e.g., [peak velocity, average velocity, etc.]
    - target_dist (float): Target distance, default is 10.0
    - output_type (str): Type of output, either "LinearModel" or "dict"
    Returns:
    If output_type is "LinearModel":
        linmod (LinearModel): Linear model structure containing
        - label: model label
        - wv0: reduced model velocity coefficients
        - wv: full model velocity coefficients
        - wr: full model duration coefficients
        - ssc: corrected PSTH for full model
        - ssc0: corrected PSTH for reduced model
        - v00: grand average peak velocity
        - r00: grand average average velocity
    If output_type is "dict":
        linmod (dict): Linear model structure containing:
        - label: model label
        - wv0: reduced model velocity coefficients
        - wv: full model velocity coefficients
        - wr: full model duration coefficients
        - ssc: corrected PSTH for full model
        - ssc0: corrected PSTH for reduced model
        - v00: grand average peak velocity
        - r00: grand average average velocity
    """

    vpeak = np.abs(params[:, 0])  # peak velocity
    sdur = np.abs(params[:, 1])  # duration
    vave = target_dist / sdur * 1e3  # average velocity
    ss = rate_matrix * 1.0  # rate matrix

    ss0 = np.nanmean(ss, axis=0)  # mean rate across trials
    dss = ss - ss0  # deviation from mean rate
    v0 = np.nanmean(vpeak)  # mean peak velocity
    r0 = np.nanmean(vave)  # mean average velocity

    dv = vpeak - v0
    dr = vave - r0
    dz = np.stack((dv, dr), axis=1)  # shape (Trial, 2)

    # Full model coefficients
    cc = dz.T @ dz
    w12 = np.linalg.pinv(cc) @ dz.T @ dss  # shape (2, Time)

    wv = w12[0, :]  # velocity coefficients
    wr = w12[1, :]  # duration coefficients

    # Reduced model with only velocity
    wv0 = (dv.T @ dss) / (dv**2).sum()  # shape (Time,)

    # Grand averages
    v00 = params0[0]
    r00 = params0[1]

    # Corrected PSTH for full model
    wc = (v00 - v0) * wv + (r00 - r0) * wr
    ssc = ss0 + wc

    # Corrected PSTH for reduced model
    wc0 = (v00 - v0) * wv0
    ssc0 = ss0 + wc0

    # here test whether output_type is LinearModel or dict
    if output_type == "LinearModel":
        linmod = LinearModel(label=label, rate=ssc, drate=w12, params0=params0)
        linmod.wv0 = wv0
        linmod.wv = wv
        linmod.wr = wr
        linmod.ssc0 = ssc0
        linmod.v00 = v00
        linmod.r00 = r00
    elif output_type == "dict":
        linmod = {
            "label": label,
            "wv0": wv0,
            "wv": wv,
            "wr": wr,
            "rate": ssc,
            "drate": w12,
            "params0": params0,
            "ssc": ssc,
            "ssc0": ssc0,
            "v00": v00,
            "r00": r00,
        }

    return linmod


def generate_linear_model(label, rate_matrix, params, params0, output_type="dict"):
    """
    Generate a linear model of the rate matrix.

    Parameters:
    - label (str): Label for the linear model
    - rate_matrix (ndarray): Rate matrix (Trial x Time)
    - params (ndarray): Parameters (peak velocity, duration, etc.) shape (Trial x n_params)
    - params0 (array-like): Grand average parameters, e.g., [peak velocity, average velocity, etc.]
    - output_type (str): Type of output, either "LinearModel" or "dict"
    Returns:
    If output_type is "LinearModel":
        linmod (LinearModel): Linear model structure containing
        - label: model label
        - rate: corrected PSTH for full model
        - drate: model coefficients for each parameter
        - params0: grand average parameters
    If output_type is "dict":
        linmod (dict): Linear model structure containing:
        - label: model label
        - rate: corrected PSTH for full model
        - drate: model coefficients for each parameter
        - params0: grand average parameters
    """

    ss = rate_matrix * 1.0  # rate matrix, multiply 1.0 for dense copy

    ss0 = np.nanmean(ss, axis=0)  # mean rate across trials
    dss = ss - ss0  # deviation from mean rate
    p0 = np.nanmean(params, axis=0)  # mean parameters

    dz = params - p0

    # Full model coefficients
    cc = dz.T @ dz
    drate = np.linalg.pinv(cc) @ dz.T @ dss  # shape (2, Time)

    # convert params0 to ndarray if not already
    if not isinstance(params0, np.ndarray):
        params0 = np.array(params0)

    # Corrected PSTH for full model
    wc = (params0 - p0) @ drate
    rate = ss0 + wc

    # here test whether output_type is LinearModel or dict
    if output_type == "LinearModel":
        linmod = LinearModel(label=label, rate=rate, drate=drate)
        linmod.params0 = params0

    elif output_type == "dict":
        linmod = {
            "label": label,
            "rate": rate,
            "drate": drate,
            "params0": params0,
        }

    return linmod
