//-------------------------------------------------------------------------
//   Copyright 2002-2023 National Technology & Engineering Solutions of
//   Sandia, LLC (NTESS).  Under the terms of Contract DE-NA0003525 with
//   NTESS, the U.S. Government retains certain rights in this software.
//
//   This file is part of the Xyce(TM) Parallel Electrical Simulator.
//
//   Xyce(TM) is free software: you can redistribute it and/or modify
//   it under the terms of the GNU General Public License as published by
//   the Free Software Foundation, either version 3 of the License, or
//   (at your option) any later version.
//
//   Xyce(TM) is distributed in the hope that it will be useful,
//   but WITHOUT ANY WARRANTY; without even the implied warranty of
//   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//   GNU General Public License for more details.
//
//   You should have received a copy of the GNU General Public License
//   along with Xyce(TM).
//   If not, see <http://www.gnu.org/licenses/>.
//-------------------------------------------------------------------------

//-------------------------------------------------------------------------
//
// Purpose        : Provide a class for more general Xyce/Alegra coupling
//
// Special Notes  : This class is meant to provide a more general Xyce/Alegra
//                  coupling API than that provided by the old N_CIR_Xygra class,
//                  which is fairly general but clearly set up and named
//                  for its primary use case, simulating coils in Alegra
//                  and coupling them loosely to Xyce.
//
// Creator        : Tom Russo, SNL, Electrical Models & Simulation
//
// Creation Date  : 2/22/2017
//
//-------------------------------------------------------------------------

#ifndef Xyce_N_CIR_ParallelXyce_H
#define Xyce_N_CIR_ParallelXyce_H

#include <map>
#include <string>

#include <N_CIR_Xyce.h>
#include <N_PDS_fwd.h>
#include <N_DEV_fwd.h>
#include <N_IO_fwd.h>
#include <N_DEV_VectorComputeInterface.h>
#include <N_IO_ExtOutInterface.h>

namespace Xyce {
namespace Circuit {

template <class ...Types>
class CallArgs
{
  
};

//-----------------------------------------------------------------------------
// Class         : ParallelSimulator
// Purpose       :
// Special Notes :
//
// Creator       : Tom Russo, SNL, Electrical and Microsystems Modeling
// Creation Date : 8/21/08
//-----------------------------------------------------------------------------
///
/// High-level Xyce interface class for use in coupling to external codes
///
/// This class is derived from the "Simulator" class of Xyce.  It provides
/// some extra functions that allow the external simulator to pass data
/// down to special interface devices
///
/// This is very similar to N_CIR_Xygra, but intended to be more general
class ParallelSimulator : public Simulator
{
public:
  using Simulator::RunStatus;

  /// Constructor
  ParallelSimulator(Xyce::Parallel::Machine comm=MPI_COMM_NULL) : Simulator(comm)
  {}

  /// Destructor
  virtual ~ParallelSimulator()
  {}

  // These are all the API calls that we are suppose to be making available
  // for external programs and/or other objects

  //---------------------------------------------------------------------------
  // Function      : setNetlistParameters
  // Purpose       : This passes a vector of pairs "key", "value" that will
  //                 be substituted during the processing of the netlist.  This
  //                 more easily allows Dakota to change any netlist parameter
  //                 during netlist setup.
  // Special Notes :
  // Scope         : public
  // Creator       : Richard Schiek, Electrical and MEMS Modeling
  // Creation Date : 10/9/2008
  //---------------------------------------------------------------------------
  void setNetlistParameters( const std::vector< std::pair< std::string, std::string > > & externalParams );


  //---------------------------------------------------------------------------
  // Function      : setNetlistParameters
  // Purpose       : Call through to the output manager to set the suffix to
  //                 be used on the output file, as in circuit + suffix + prn
  //                 This is useful in Dakota controlled runs to keep each
  //                 simulation from overwritting the last one.
  // Special Notes :
  // Scope         : public
  // Creator       : Richard Schiek, Electrical and MEMS Modeling
  // Creation Date : 10/9/2008
  //---------------------------------------------------------------------------
  void setOutputFileSuffix( const std::string newSuffix );

  //---------------------------------------------------------------------------
  // Function      : run
  // Purpose       :
  // Special Notes :
  // Scope         : public
  // Creator       : Eric Keiter, SNL, Parallel Computational Sciences
  // Creation Date : 02/19/01
  //---------------------------------------------------------------------------
  RunStatus run(int argc, char **argv);

  //---------------------------------------------------------------------------
  // Function      : initialize
  // Purpose       : To initialize Xyce to be driven by the SAX
  //                 simulation backplane. This includes the following:
  //                    Set up and register the Parallel Manager.
  //                    Parse the command-line arguments.
  //                    Redirect the output stream of processor 1,
  //                    if requested.
  //                    Read in the Netlist.
  //                    Allocate and register the external packages.
  //                    Set up the representation of the circuit topology.
  //                    Set up the matrix structures.
  //                    Initialize the solvers.
  // Special Notes :
  // Scope         : public
  // Creator       : Lon Waters, Lisa Renee Maynes
  // Creation Date : 05/28/03
  //---------------------------------------------------------------------------
  RunStatus initialize(int argc, char **argv);

  using Simulator::initializeEarly;
  using Simulator::initializeLate;

  //---------------------------------------------------------------------------
  // Function      : runSimulation
  // Purpose       :
  // Special Notes :
  // Scope         : public
  // Creator       : Eric Keiter, SNL, Parallel Computational Sciences
  // Creation Date : 5/27/00
  //---------------------------------------------------------------------------
  RunStatus runSimulation();
  RunStatus runWorker();

  bool getDeviceNames(const std::string & modelGroupName, std::vector<std::string> & deviceNames);
  bool getAllDeviceNames(std::vector<std::string> & deviceNames);
  bool getDeviceParamVal(const std::string full_param_name, double& val) const;
  bool checkDeviceParamName(const std::string full_param_name) const;

  bool getAdjGIDsForDevice(const std::string deviceName, std::vector<int> & adj_GIDs) const;
  bool getNumAdjNodesForDevice(const std::string deviceName, int & numAdjNodes) const;

  //---------------------------------------------------------------------------
  // Function      : getDACDeviceNames
  // Purpose       : Gets the (stripped) names of the DAC devices
  //                 in the circuit.
  // Special Notes :
  // Scope         : public
  // Creator       : Lon Waters, Lisa Renee Maynes
  // Creation Date : 06/13/03
  //---------------------------------------------------------------------------
  bool getDACDeviceNames(std::vector< std::string >& dacNames);

  //---------------------------------------------------------------------------
  // Function      : getADCMap
  // Purpose       : Gets the (stripped) names of the ADC devices
  //                 in the circuit(as key of map) and map of parameters
  //                 (keyed by parameter name) for each device
  // Special Notes :
  // Scope         : public
  // Creator       : Tom Russo, SNL, Component Information and Models
  // Creation Date : 05/06/2004
  //---------------------------------------------------------------------------
  bool getADCMap(std::map<std::string,std::map<std::string,double> >& ADCMap);

  //---------------------------------------------------------------------------
  // Function      : updateTimeVoltagePairs
  // Purpose       : Update the DAC devices in a circuit by adding the set
  //                 of time and voltage pairs built up on the "digital side"
  //                 since the last update and by removing the time-voltage
  //                 pairs for times that pre-date the given simulation time.
  // Special Notes :
  // Scope         : public
  // Creator       : Lon Waters, Lisa Renee Maynes
  // Creation Date : 06/10/03
  //---------------------------------------------------------------------------
  bool updateTimeVoltagePairs(
        std::map< std::string, std::vector< std::pair<double,double> >* > const&
        timeVoltageUpdateMap);

  //---------------------------------------------------------------------------
  // Function      : getTimeVoltagePairs
  // Purpose       : query the DAC devices in a circuit for the set
  //                 of time and voltage pairs
  // Special Notes : Calling this function clears the ADC devices time-voltage
  //                 pair vector.
  // Scope         : public
  // Creator       : Tom Russo, SNL ComponentInformation and Models
  // Creation Date : 05/10/2004
  //---------------------------------------------------------------------------
  bool getTimeVoltagePairs(
        std::map< std::string, std::vector< std::pair<double,double> > > &
        timeVoltageUpdateMap);

  //---------------------------------------------------------------------------
  // Function      : getTimeVoltagePairsSz
  // Purpose       : Returns the largest size of the TV Pairs data in the ADCs
  // Special Notes :
  // Scope         : public
  // Creator       :
  // Creation Date : 11/11/2021
  //---------------------------------------------------------------------------
  bool getTimeVoltagePairsSz(int &maximumSize);

  //---------------------------------------------------------------------------
  // Function      : getTimeStatePairs
  // Purpose       : query the DAC devices in a circuit for the set
  //                 of time and state pairs
  // Special Notes :
  // Scope         : public
  // Creator       : Pete Sholander, SNL
  // Creation Date : 11/13/2018
  //---------------------------------------------------------------------------
  bool getTimeStatePairs(
        std::map< std::string, std::vector< std::pair<double,int> > > &
        timeStateUpdateMap);

  //----------------------------------------------------------------------------
  // Function       : setADCWidths
  // Purpose        : Update the ADC devices in a circuit by informing them
  //                  of the width of their bitvector output on the
  //                  "digital side"
  // Special Notes  :
  // Scope          :
  // Creator        : Tom Russo
  // Creation Date  : 05/07/2004
  //----------------------------------------------------------------------------
  bool setADCWidths(std::map< std::string, int > const& ADCWidthMap);

  //----------------------------------------------------------------------------
  // Function       : getADCWidth
  // Purpose        : get the width of the specified ADC devices bitvector output
  //                  on the "digital side"
  // Special Notes  :
  // Scope          :
  // Creator        : Pete Sholander
  // Creation Date  : 11/16/2018
  //----------------------------------------------------------------------------
  bool getADCWidths(std::map< std::string, int > & ADCWidthMap);

  //---------------------------------------------------------------------------
  // Function      : simulateUntil
  // Purpose       : To continue the existing analog circuit simulation
  //                 until either the given <requestedUntilTime> is reached
  //                 or the simulation termination criterion is met.
  //                 Return a Boolean indicating whether the simulation
  //                 run was successful. (Note that the run is successful
  //                 even when the given <requestedUntilTime> is not reached,
  //                 so long as the run completed normally.)
  // Special Notes : The time variables are in units of seconds.
  // Scope         : public
  // Creator       : Lon Waters, Lisa Renee Maynes
  // Creation Date : 05/28/03
  //---------------------------------------------------------------------------
  bool simulateUntil(double requestedUntilTime, double& completedUntilTime);

  //---------------------------------------------------------------------------
  // Function      : finalize
  // Purpose       : To clean up after driving Xyce with the SIMBUS
  //                 simulation backplane. This includes the following:
  //                    Free any dynamically allocated memory...
  // Special Notes :
  // Scope         : public
  // Creator       : Lon Waters, Lisa Renee Maynes
  // Creation Date : 05/29/03
  //---------------------------------------------------------------------------
  RunStatus finalize();

  using Simulator::reportTotalElapsedTime;

  //
  // checkes if a given name, variable_name, exists in as the name of a measure
  // with the measure manager.
  //
  bool checkResponseVar(const std::string &variable_name) const;
  //
  // if the varaible_name exists in the measure manager, then return its
  // value in result.
  //
  bool obtainResponse(const std::string& variable_name, double &result) const;

  //
  // checks that the given parameter exists.
  // return true if it does and false otherwise.
  bool checkCircuitParameterExists(std::string paramName);

  //
  // sets the given parameter through the device manager.
  // returns false if the parameter did not exist (and thus was not set)
  bool setCircuitParameter(std::string paramName, double paramValue);

  //
  // gets a value from the current simulation based on the name passed it
  // The name can be parameter name, voltage node or current (ie. solution variable)
  // or a measure name.  return false if the value was not found.
  bool getCircuitValue(std::string paramName, double& paramValue);

  //
  // accessor to AnalysisManger function.
  // for non-time based analysis it returns zero.
  //
  double getTime();

  //
  // accessor to AnalysisManger function.
  // for non-time based analysis it returns zero.
  //
  double getFinalTime();

  // report on whether simulation is finished or not
  bool simulationComplete();

protected:
  enum WorkerCommand
  {
    INITIALIZE,
    RUN_SIMULATION,
    SIMULATE_UNTIL,
    SIMULATION_COMPLETE,
    CHECK_CIRCUIT_PARAMETER_EXISTS,
    GET_TIME,
    GET_FINAL_TIME,
    GET_DEVICE_NAMES,
    GET_ALL_DEVICE_NAMES,
    GET_DAC_DEVICE_NAMES,
    CHECK_DEVICE_PARAM_NAME,
    GET_DEVICE_PARAM_VAL,
    GET_NUM_ADJ_NODES_FOR_DEVICE,
    GET_ADJ_GIDS_FOR_DEVICE,
    GET_ADC_MAP,
    UPDATE_TIME_VOLTAGE_PAIRS,
    GET_TIME_VOLTAGE_PAIRS,
    GET_TIME_VOLTAGE_PAIRS_SZ,
    GET_TIME_STATE_PAIRS,
    SET_ADC_WIDTHS,
    GET_ADC_WIDTHS,
    GET_CIRCUIT_VALUE,
    SET_CIRCUIT_PARAMETER,
    CHECK_RESPONSE_VAR,
    OBTAIN_RESPONSE,
    FINALIZE
  };

};

} // namespace Circuit
} // namespace Xyce

#endif // Xyce_N_CIR_ParallelXyce_H
